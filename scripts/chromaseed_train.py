"""Staged, hash-bound palette pretraining and original-TRAIN skin adaptation."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from chromaseed import (
    X_MEAN,
    X_STD,
    Y_MEAN,
    Y_STD,
    ChromaSeed,
    make_palette,
    pack_model,
    predict,
    unpack_model,
)
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    load_model,
    metrics,
    roles,
    sha,
    synchronize,
    weights_for,
    write_json,
)

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
ARMS = ("scratch", "clean_palette", "rendered_palette", "shuffled_palette", "skin_long")
PRETRAINED = ("clean_palette", "rendered_palette", "shuffled_palette")
LRS = (.0003, .001, .003)
CHECKPOINTS = (64, 256, 1024)
PRETRAIN_STEPS = 2048
PROTOCOL = ROOT / "docs/research/chromaseed_v1_protocol.md"


def train_trace(initial, x, y, weights, seed, lr, steps, checkpoints, device):
    weights = np.asarray(weights, dtype=np.float32)
    if x.shape != (len(y), 36) or y.shape != (len(x), 3) or weights.shape != (len(x),):
        raise ValueError("Invalid training dimensions")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Training weights must be finite and positive")
    if not checkpoints or any(k < 1 or k > steps for k in checkpoints) or steps not in checkpoints:
        raise ValueError("Checkpoints must include the last update")
    synchronize(device)
    start = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    net = unpack_model(initial, device).train()
    xt = torch.as_tensor((np.asarray(x, np.float32) - X_MEAN) / X_STD, device=device)
    yt = torch.as_tensor((np.asarray(y, np.float32) - Y_MEAN) / Y_STD, device=device)
    wt = torch.as_tensor(weights, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 19001)
    indices = torch.multinomial(wt, steps * 256, replacement=True, generator=generator).reshape(steps, 256)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=.01)
    snapshots, trace = {}, {}
    for step in range(1, steps + 1):
        idx = indices[step - 1]
        optimizer.zero_grad(set_to_none=True)
        loss = (net(xt[idx]) - yt[idx]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        if step in checkpoints:
            synchronize(device)
            elapsed = time.perf_counter() - start
            with torch.no_grad():
                errors = (net(xt) - yt).square().mean(1)
                mean_error = float((errors * wt).sum() / wt.sum())
            snapshots[step] = pack_model(net)
            trace[str(step)] = {"seconds": elapsed, "weighted_normalized_mse": mean_error}
    synchronize(device)
    receipt = {"steps": steps, "batch_size": 256, "lr": lr, "seed": seed,
               "fit_seconds": time.perf_counter() - start, "checkpoints": trace,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
               "n_fit_rows": len(x)}
    return snapshots, receipt


def write_trace(folder, models, receipt):
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(receipt)
    record["artifacts"] = {}
    for step, model in models.items():
        path = folder / f"step{step}.npz"
        np.savez(path, **model)
        record["artifacts"][str(step)] = {"sha256": sha(path), "serialized_bytes": path.stat().st_size,
                                        "numeric_bytes": sum(v.nbytes for v in model.values() if v.dtype.kind == "f")}
    write_json(folder / "trace.json", record)
    return record


def read_trace(folder):
    record = json.loads((folder / "trace.json").read_text(encoding="utf-8"))
    models = {}
    for step, artifact in record["artifacts"].items():
        path = folder / f"step{step}.npz"
        if sha(path) != artifact["sha256"]:
            raise ValueError("Frozen trace artifact changed")
        models[int(step)] = load_model(path)
    return models, record


def obtain_trace(folder, initial, x, y, weights, seed, lr, steps, checkpoints, device):
    if (folder / "trace.json").exists():
        models, record = read_trace(folder)
        if record["seed"] != seed or record["lr"] != lr or record["steps"] != steps or record["n_fit_rows"] != len(x):
            raise ValueError("Existing trace configuration mismatch")
        return models, record
    models, record = train_trace(initial, x, y, weights, seed, lr, steps, checkpoints, device)
    return models, write_trace(folder, models, record)


def palette_metrics(model, x, y):
    de = delta_e00(predict(model, x), y)
    return {"mean_delta_e00": float(de.mean()), "median_delta_e00": float(np.median(de)),
            "p90_delta_e00": float(np.quantile(de, .9)), "n_synthetic_colors": len(de)}


def lock_sources(args):
    if not args.cache.is_absolute() or sha(args.cache) != CACHE_HASH:
        raise ValueError("Expected explicit hash-verified original TRAIN path")
    files = [Path(__file__), ROOT / "scripts/chromaseed.py", PROTOCOL,
             ROOT / "scripts/skin_local_search_train.py", ROOT / "scripts/skin_local_search_core.py",
             ROOT / "src/luma_skin_vision/color.py"]
    lock = {"cache_sha256": CACHE_HASH, "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "device": args.device, "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "seeds": list(SEEDS), "lrs": list(LRS), "checkpoints": list(CHECKPOINTS),
            "pretrain_steps": PRETRAIN_STEPS, "batch_size": 256}
    path = args.run / "source_lock.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Source/config binding changed; use a new run directory")
    else:
        write_json(path, lock)


def prepare_palette(run):
    receipt_path = run / "palette/receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for name in ("train", "validation"):
            if sha(run / "palette" / (name + ".npz")) != receipt[name]["sha256"]:
                raise ValueError("Synthetic palette changed")
        return receipt
    start = time.perf_counter()
    folder = run / "palette"
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {"evidence": "SYNTHETIC local color surfaces, not measured skin"}
    for name, count, seed in (("train", 32768, 731013), ("validation", 4096, 731014)):
        data = make_palette(count, seed)
        path = folder / (name + ".npz")
        np.savez(path, **data)
        receipt[name] = {"n_colors": count, "seed": seed, "sha256": sha(path),
                         "clipped_observation_color_fraction": float(np.mean(np.any((data["observed_rgb"] <= 0) | (data["observed_rgb"] >= 1), axis=1)))}
    receipt["generation_seconds"] = time.perf_counter() - start
    write_json(receipt_path, receipt)
    return receipt


def pretrain_phase(run, device):
    palette_receipt = prepare_palette(run)
    with np.load(run / "palette/train.npz", allow_pickle=False) as archive:
        training = {k: archive[k] for k in ("clean", "rendered", "target")}
    with np.load(run / "palette/validation.npz", allow_pickle=False) as archive:
        validation = {k: archive[k] for k in ("clean", "rendered", "target")}
    manifest = {"palette_receipt_sha256": sha(run / "palette/receipt.json"),
                "palette_generation_seconds": palette_receipt["generation_seconds"], "models": [], "initial_hashes": {}}
    for seed in SEEDS:
        initial_path = run / "pretrain" / f"initial_s{seed}.npz"
        torch.manual_seed(seed)
        initial = pack_model(ChromaSeed())
        if initial_path.exists():
            stored = load_model(initial_path)
            for key in initial:
                np.testing.assert_array_equal(initial[key], stored[key])
        else:
            initial_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(initial_path, **initial)
        manifest["initial_hashes"][str(initial_path.relative_to(run))] = sha(initial_path)
        for arm in PRETRAINED:
            x = training["clean" if arm == "clean_palette" else "rendered"]
            target = training["target"]
            if arm == "shuffled_palette":
                target = target[np.random.default_rng(seed + 88342).permutation(len(target))]
            folder = run / "pretrain" / f"{arm}_s{seed}"
            models, record = obtain_trace(folder, initial, x, target, np.ones(len(x)), seed, .001,
                                          PRETRAIN_STEPS, (256, 1024, PRETRAIN_STEPS), device)
            result = {"arm": arm, "seed": seed, "trace_sha256": sha(folder / "trace.json"),
                      "fit_seconds": record["fit_seconds"], "synthetic_validation": {
                          style: palette_metrics(models[PRETRAIN_STEPS], validation[style], validation["target"])
                          for style in ("clean", "rendered")}}
            manifest["models"].append(result)
            print(json.dumps({"stage": "pretrain", **result}), flush=True)
    write_json(run / "pretrain_manifest.json", manifest)
    print("Synthetic pretraining frozen. No real skin labels used by this stage.", flush=True)


def verify_pretraining(run):
    manifest = json.loads((run / "pretrain_manifest.json").read_text(encoding="utf-8"))
    if sha(run / "palette/receipt.json") != manifest["palette_receipt_sha256"]:
        raise ValueError("Palette receipt changed")
    for path, digest in manifest["initial_hashes"].items():
        if sha(run / path) != digest:
            raise ValueError("Initial weights changed")
    for row in manifest["models"]:
        folder = run / "pretrain" / f"{row['arm']}_s{row['seed']}"
        if sha(folder / "trace.json") != row["trace_sha256"]:
            raise ValueError("Pretraining trace changed")
        read_trace(folder)
    return manifest


def adaptation_initial(arm, seed, x, y, person, site, base_folder, run, device):
    if arm in PRETRAINED:
        models, record = read_trace(run / "pretrain" / f"{arm}_s{seed}")
        return models[PRETRAIN_STEPS], record["fit_seconds"]
    initial = load_model(run / "pretrain" / f"initial_s{seed}.npz")
    if arm == "scratch":
        return initial, 0.0
    models, record = obtain_trace(base_folder / f"warmup_s{seed}", initial, x, y, weights_for(person, site),
                                  seed, .001, PRETRAIN_STEPS, (PRETRAIN_STEPS,), device)
    return models[PRETRAIN_STEPS], record["fit_seconds"]


def fit_phase(data, run, device):
    if not (run / "pretrain_manifest.json").exists():
        raise ValueError("Complete synthetic pretraining first")
    verify_pretraining(run)
    all_choices, inner_records = {}, []
    for protocol, (fit, held) in roles(data["patient"], data["device"]).items():
        x, y = data["color"][fit], data["target"][fit]
        p, s, c = data["patient"][fit], data["site"][fit], data["device"][fit]
        folds = folds_for(p, c)
        folder = run / protocol
        folder.mkdir(parents=True, exist_ok=True)
        role_path = folder / "roles.npz"
        if role_path.exists():
            with np.load(role_path, allow_pickle=False) as previous:
                for key, value in (("fit", fit), ("held", held), ("folds", folds)):
                    np.testing.assert_array_equal(previous[key], value)
        else:
            np.savez(role_path, fit=fit, held=held, folds=folds)
        choices = {}
        for arm in ARMS:
            config_scores, config_curves = [], []
            for config, lr in enumerate(LRS):
                oof = {seed: {step: np.empty_like(y) for step in CHECKPOINTS} for seed in SEEDS}
                for seed in SEEDS:
                    for fold in range(3):
                        a, b = folds != fold, folds == fold
                        base_folder = folder / "inner" / f"fold{fold}"
                        initial, inherited_seconds = adaptation_initial(arm, seed, x[a], y[a], p[a], s[a], base_folder, run, device)
                        trace_folder = base_folder / f"{arm}_c{config}_s{seed}"
                        models, record = obtain_trace(trace_folder, initial, x[a], y[a], weights_for(p[a], s[a]),
                                                      seed, lr, 1024, CHECKPOINTS, device)
                        fold_record = {"protocol": protocol, "arm": arm, "seed": seed, "fold": fold, "lr": lr,
                                       "adapt_seconds": record["fit_seconds"], "inherited_fit_seconds": inherited_seconds,
                                       "peak_allocated_bytes": record["peak_allocated_bytes"], "curve": {}}
                        for step in CHECKPOINTS:
                            pred = predict(models[step], x[b])
                            oof[seed][step][b] = pred
                            fold_record["curve"][str(step)] = metrics(pred, y[b], p[b], s[b])
                        inner_records.append(fold_record)
                curve = {str(step): float(np.mean([metrics(oof[seed][step], y, p, s)["person_mean"] for seed in SEEDS]))
                         for step in CHECKPOINTS}
                config_scores.append(curve["1024"])
                config_curves.append(curve)
                print(json.dumps({"stage": "inner", "protocol": protocol, "arm": arm, "lr": lr, "curve": curve}), flush=True)
            best = int(np.argmin(config_scores))
            choices[arm] = {"config": best, "lr": LRS[best], "primary_steps": 1024,
                            "inner_person_mean": config_scores[best], "config_scores": config_scores,
                            "chosen_lr_inner_curve": config_curves[best]}
        all_choices[protocol] = choices
        write_json(folder / "selection.json", choices)
        for arm, choice in choices.items():
            for seed in SEEDS:
                initial, inherited_seconds = adaptation_initial(arm, seed, x, y, p, s, folder / "final", run, device)
                trace_folder = folder / "final" / f"{arm}_s{seed}"
                _, record = obtain_trace(trace_folder, initial, x, y, weights_for(p, s), seed, choice["lr"],
                                         1024, CHECKPOINTS, device)
                write_json(trace_folder / "lineage.json", {"inherited_fit_seconds": inherited_seconds,
                                                           "adapt_seconds": record["fit_seconds"],
                                                           "trace_sha256": sha(trace_folder / "trace.json")})
    write_json(run / "inner_records.json", inner_records)
    manifest = {"pretrain_manifest_sha256": sha(run / "pretrain_manifest.json"), "protocols": all_choices,
                "final_trace_hashes": {str(path.relative_to(run)): sha(path) for path in sorted(run.glob("*/final/*/trace.json"))},
                "role_hashes": {str(path.relative_to(run)): sha(path) for path in sorted(run.glob("*/roles.npz"))}}
    write_json(run / "frozen_selections.json", manifest)
    print("All ChromaSeed choices and final checkpoints frozen; no outer evaluation in this phase.", flush=True)


def evaluate_phase(data, run):
    frozen_path = run / "frozen_selections.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if sha(run / "pretrain_manifest.json") != frozen["pretrain_manifest_sha256"]:
        raise ValueError("Pretraining manifest changed")
    verify_pretraining(run)
    for key in ("final_trace_hashes", "role_hashes"):
        for path, digest in frozen[key].items():
            if sha(run / path) != digest:
                raise ValueError("Frozen final lineage changed")
    # Preflight every deployed checkpoint before computing outer errors.
    models_by_role = {}
    for protocol, choices in frozen["protocols"].items():
        for arm in choices:
            for seed in SEEDS:
                models_by_role[(protocol, arm, seed)] = read_trace(run / protocol / "final" / f"{arm}_s{seed}")
    records, zero_shot = [], []
    for protocol, choices in frozen["protocols"].items():
        with np.load(run / protocol / "roles.npz", allow_pickle=False) as role_data:
            held = role_data["held"]
        x, y = data["color"][held], data["target"][held]
        p, s, c = data["patient"][held], data["site"][held], data["device"][held]
        for arm, choice in choices.items():
            for seed in SEEDS:
                models, trace = models_by_role[(protocol, arm, seed)]
                lineage = json.loads((run / protocol / "final" / f"{arm}_s{seed}/lineage.json").read_text(encoding="utf-8"))
                for step in CHECKPOINTS:
                    model = models[step]
                    pred = predict(model, x)
                    record = {"protocol": protocol, "arm": arm, "seed": seed, "step": step, "lr": choice["lr"],
                              "metrics": metrics(pred, y, p, s, c), "adapt_seconds": trace["checkpoints"][str(step)]["seconds"],
                              "inherited_fit_seconds": lineage["inherited_fit_seconds"],
                              "numeric_bytes": trace["artifacts"][str(step)]["numeric_bytes"],
                              "serialized_bytes": trace["artifacts"][str(step)]["serialized_bytes"]}
                    records.append(record)
                    np.savez(run / protocol / "final" / f"{arm}_s{seed}" / f"outer{step}.npz", prediction=pred, target=y,
                             person=p, site=s, camera=c)
                print(json.dumps({"stage": "outer", "protocol": protocol, "arm": arm, "seed": seed,
                                  "person_delta_e00": records[-1]["metrics"]["person_mean"]}), flush=True)
        for arm in PRETRAINED:
            for seed in SEEDS:
                models, _ = read_trace(run / "pretrain" / f"{arm}_s{seed}")
                zero_shot.append({"protocol": protocol, "arm": arm, "seed": seed,
                                  "metrics": metrics(predict(models[PRETRAIN_STEPS], x), y, p, s, c)})
    write_json(run / "evaluation.json", {"evidence": "Exploratory reused original-TRAIN roles, overlapping protocols",
                                          "frozen_selection_sha256": sha(frozen_path), "models": records,
                                          "zero_shot_diagnostic": zero_shot})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "experiments/runs/chromaseed_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("pretrain", "fit", "evaluate"), required=True)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    lock_sources(args)
    start = time.perf_counter()
    if args.stage == "pretrain":
        pretrain_phase(args.run, args.device)
    else:
        with np.load(args.cache, allow_pickle=False) as archive:
            data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
        if data["color"].shape != (966, 36):
            raise ValueError("Unexpected real source shape")
        if args.stage == "fit":
            fit_phase(data, args.run, args.device)
        else:
            evaluate_phase(data, args.run)
    write_json(args.run / f"{args.stage}_completion.json", {"seconds_this_invocation": time.perf_counter() - start,
                                                            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
