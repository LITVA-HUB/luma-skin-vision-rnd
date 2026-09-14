"""Resumable TRAIN-only ChromaSeed-R experiment; separate selection/evaluation."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chromaseed_refine import (
    FAMILIES,
    BankAdamW,
    BankNet,
    apply_exit_policy,
    fit_preprocessor,
    transform,
)
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    synchronize,
    weights_for,
    write_json,
)

SEEDS = (17, 29, 43)
LRS = (.0003, .001, .003)
CHECKPOINTS = (512, 2048, 8192)
THRESHOLDS = (0., .25, .5, 1.)
SLOTS = tuple((seed, lr) for seed in SEEDS for lr in LRS)
_WARMUP_STREAM = None
BOUND_SOURCES = (
    "scripts/chromaseed_refine.py", "scripts/chromaseed_refine_train.py",
    "scripts/skin_local_search_train.py", "scripts/skin_local_search_core.py",
    "src/luma_skin_vision/color.py", "docs/research/chromaseed_refine_v1_protocol.md",
)


def setup(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def source_lock(run, cache, device):
    source = {p: sha(ROOT / p) for p in BOUND_SOURCES}
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("only the immutable original TRAIN cache is allowed")
    lock = {"schema": 1, "sources": source, "cache_sha256": CACHE_HASH,
            "families": list(FAMILIES), "seeds": list(SEEDS), "lrs": list(LRS),
            "checkpoints": list(CHECKPOINTS), "thresholds": list(THRESHOLDS),
            "batch_size": 64, "dtype": "float32", "tf32": False,
            "device": device, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "evidence": "historically exposed TRAIN-only exploratory roles"}
    path = run / "source_lock.json"
    if path.exists():
        previous = read_json(path)
        if previous != lock:
            raise ValueError("source/config/environment changed after freeze; create a distinct run")
    else:
        write_json(path, lock)
    return sha(path)


def load_data(cache):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("unexpected data cache")
    with np.load(cache, allow_pickle=False) as z:
        data = {key: z[key] for key in ("color", "tokens", "target", "patient", "site", "device")}
    if data["color"].shape != (966, 36) or data["tokens"].shape != (966, 64, 18):
        raise ValueError("unexpected feature shape")
    return data


def sampling_indices(weights, seeds, steps, batch_size=64):
    probability = np.asarray(weights, np.float64)
    if len(probability) == 0 or not np.isfinite(probability).all() or np.any(probability <= 0):
        raise ValueError("sampling weights must be finite and positive")
    probability = probability / probability.sum()
    sequences = {}
    for seed in set(seeds):
        rng = np.random.default_rng(seed + 9001)
        sequences[seed] = rng.choice(len(weights), (steps, batch_size), p=probability)
    return np.stack([sequences[seed] for seed in seeds])


def train_bank(x, patches, y, weights, family, slots, steps, checkpoints, device, progress=None, engine="eager"):
    global _WARMUP_STREAM
    if steps <= 0 or not checkpoints or tuple(sorted(set(checkpoints))) != tuple(checkpoints) or checkpoints[-1] != steps or checkpoints[0] <= 0:
        raise ValueError("checkpoints must be unique, increasing and end at positive steps")
    if not slots:
        raise ValueError("at least one independent training slot is required")
    if engine not in ("eager", "cuda_graph") or (engine == "cuda_graph" and not str(device).startswith("cuda")):
        raise ValueError("CUDA graph engine requires a CUDA device")
    synchronize(device)
    started = time.perf_counter()
    prep = fit_preprocessor(x, patches, y, weights)
    xn, tn, base = transform(prep, x, patches)
    yn = (y.astype(np.float32) - prep["y_mean"]) / prep["y_std"]
    tensors = [torch.as_tensor(a, device=device, dtype=torch.float32) for a in (xn, tn, base, yn)]
    xt, tt, bt, yt = tensors
    seeds, lrs = zip(*slots, strict=True)
    net = BankNet(family, seeds).to(device)
    optimizer = BankAdamW(net.parameters(), lrs)
    indices = torch.as_tensor(sampling_indices(weights, seeds, steps), device=device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    def iteration(index, corrections=None):
        net.zero_grad(set_to_none=True)
        output, _, penalty = net(xt[index], tt[index], bt[index])
        per_pass = (output - yt[index].unsqueeze(-2)).square().mean((1, 3))
        loss = .6 * per_pass.mean(-1) + .4 * per_pass[:, -1]
        if family == "recur_dynamic":
            loss = loss + .001 * penalty
        loss.sum().backward()
        optimizer.step(corrections)
        return loss

    graph = None
    if engine == "cuda_graph":
        # GPU counter indexes precomputed sample sequences and exact Python-computed
        # bias corrections. Replays do not reuse step-one Adam bias corrections.
        correction_table = torch.as_tensor(np.array([[1. - .9 ** t, np.sqrt(1. - .999 ** t)]
                                                     for t in range(1, steps + 1)], np.float32), device=device)
        counter = torch.zeros(1, dtype=torch.int64, device=device)

        def captured_iteration():
            index = indices.index_select(1, counter).squeeze(1)
            corrections = correction_table.index_select(0, counter)[0]
            current_loss = iteration(index, corrections)
            counter.add_(1)
            return current_loss

        initial = net.theta.detach().clone()
        # Reuse a stream: cuBLAS workspaces are cached per stream and otherwise
        # accumulate across a long bank study despite each model being tiny.
        if _WARMUP_STREAM is None:
            _WARMUP_STREAM = torch.cuda.Stream()
        stream = _WARMUP_STREAM
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(min(3, steps)):
                captured_iteration()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad():
            net.theta.copy_(initial)
            counter.zero_()
            for moment in optimizer.m + optimizer.v:
                moment.zero_()
        net.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_loss = captured_iteration()
        del initial
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    loop_start = time.perf_counter()
    models, trace = {}, []
    for step in range(1, steps + 1):
        if graph is None:
            loss = iteration(indices[:, step - 1])
        else:
            graph.replay()
            loss = captured_loss
        if step in checkpoints:
            models[step] = {**prep, "theta": net.theta.detach().cpu().numpy().copy(), "family": np.asarray(family)}
        if step % 512 == 0 or step == steps:
            synchronize(device)
            entry = {"step": step, "seconds": time.perf_counter() - loop_start,
                     "sampled_objective_by_slot": loss.detach().cpu().tolist()}
            if not np.isfinite(entry["sampled_objective_by_slot"]).all():
                raise FloatingPointError("nonfinite objective; study must retain failed configuration")
            trace.append(entry)
            if progress is not None:
                progress(entry)
    synchronize(device)
    elapsed = time.perf_counter() - started
    receipt = {"steps": steps, "checkpoints": list(checkpoints), "batch_size": 64, "engine": engine,
               "slots": [{"seed": s, "lr": lr} for s, lr in slots], "n_fit_images": len(x),
               "family": family, "learned_parameters_per_model": net.theta.shape[1],
               "normalizer_and_anchor_floats": sum(a.size for a in prep.values()),
               "setup_seconds": setup_seconds, "bank_total_seconds": elapsed,
               "amortized_seconds_per_fit": elapsed / len(slots), "trace": trace,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None}
    return models, receipt


@torch.no_grad()
def predict_bank(payload, x, patches, device, chunk_size=128):
    xn, tn, base = transform(payload, x, patches)
    theta = payload["theta"]
    net = BankNet(str(payload["family"]), [17] * len(theta)).to(device).eval()
    net.theta.copy_(torch.as_tensor(theta, device=device))
    predictions, connections = [], []
    for start in range(0, len(x), chunk_size):
        end = start + chunk_size
        tensors = [torch.as_tensor(a[start:end], device=device).unsqueeze(0).expand(len(theta), *a[start:end].shape)
                   for a in (xn, tn, base)]
        outputs, counts, _ = net(*tensors)
        predictions.append(outputs.cpu().numpy() * payload["y_std"] + payload["y_mean"])
        connections.append(counts.cpu().numpy())
    return np.concatenate(predictions, axis=1), np.concatenate(connections, axis=1)


def completed_bank(directory, lock_hash):
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        return False
    receipt = read_json(receipt_path)
    if receipt["source_lock_sha256"] != lock_hash:
        raise ValueError("bank source-lock mismatch")
    for name, digest in receipt["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("completed bank artifact changed")
    return True


def progress_callback(run, bank_name, bank_number, total_banks):
    def emit(entry):
        write_json(run / "progress.json", {"status": "running", "pid": os.getpid(),
                    "bank": bank_name, "bank_number": bank_number, "total_banks": total_banks,
                    "step": entry["step"], "bank_seconds": entry["seconds"],
                    "updated_unix": time.time()})
        if entry["step"] in CHECKPOINTS:
            print(json.dumps({"bank": bank_name, "step": entry["step"], "seconds": round(entry["seconds"], 2)}, ensure_ascii=False), flush=True)
    return emit


def inner_stage(run, data, device, lock_hash):
    all_roles = roles(data["patient"], data["device"])
    number = 0
    for role, (outer_fit, _) in all_roles.items():
        fit_indices = np.flatnonzero(outer_fit)
        folds = folds_for(data["patient"][outer_fit], data["device"][outer_fit])
        for family in FAMILIES:
            for fold in range(3):
                number += 1
                bank_name = f"inner/{role}/{family}/fold{fold}"
                directory = run / bank_name
                if completed_bank(directory, lock_hash):
                    continue
                fit_idx, held_idx = fit_indices[folds != fold], fit_indices[folds == fold]
                if set(data["patient"][fit_idx]) & set(data["patient"][held_idx]):
                    raise ValueError("person leakage")
                print(f"START {number}/45 {bank_name}", flush=True)
                weight = weights_for(data["patient"][fit_idx], data["site"][fit_idx])
                models, receipt = train_bank(data["color"][fit_idx], data["tokens"][fit_idx], data["target"][fit_idx],
                    weight, family, SLOTS, CHECKPOINTS[-1], CHECKPOINTS, device,
                    progress_callback(run, bank_name, number, 45), engine="cuda_graph" if device.startswith("cuda") else "eager")
                files = {}
                for step, payload in models.items():
                    model_path = directory / f"bank_{step}.npz"
                    atomic_npz(model_path, **payload)
                    predictions, counts = predict_bank(payload, data["color"][held_idx], data["tokens"][held_idx], device)
                    prediction_path = directory / f"oof_{step}.npz"
                    atomic_npz(prediction_path, row_indices=held_idx, predictions=predictions, connections=counts)
                    files[model_path.name], files[prediction_path.name] = sha(model_path), sha(prediction_path)
                receipt.update({"source_lock_sha256": lock_hash, "role": role, "fold": fold,
                                "n_fit_people": len(np.unique(data["patient"][fit_idx])),
                                "n_held_people": len(np.unique(data["patient"][held_idx])), "files": files})
                write_json(directory / "receipt.json", receipt)
    write_json(run / "inner_complete.json", {"source_lock_sha256": lock_hash, "banks": 45, "independent_training_traces": 405})


def read_oof(run, role, family, checkpoint, expected_indices):
    rows, outputs, connections = [], [], []
    for fold in range(3):
        with np.load(run / "inner" / role / family / f"fold{fold}" / f"oof_{checkpoint}.npz", allow_pickle=False) as z:
            rows.append(z["row_indices"])
            outputs.append(z["predictions"])
            connections.append(z["connections"])
    rows = np.concatenate(rows)
    order = np.argsort(rows)
    if not np.array_equal(rows[order], expected_indices):
        raise ValueError("OOF rows missing, duplicated or out of role")
    return np.concatenate(outputs, axis=1)[:, order], np.concatenate(connections, axis=1)[:, order]


def seed_averaged_policy(outputs, threshold, idx, data):
    records, steps, per_slot = [], [], []
    for prediction in outputs:
        selected, count = apply_exit_policy(prediction, threshold)
        records.append(metrics(selected, data["target"][idx], data["patient"][idx], data["site"][idx]))
        steps.append(float(count.mean()))
        per_slot.append(selected)
    return {"person_mean": float(np.mean([m["person_mean"] for m in records])),
            "p90": float(np.mean([m["p90"] for m in records])),
            "mean_passes": float(np.mean(steps)), "seed_metrics": records}, np.stack(per_slot)


def select_stage(run, data, lock_hash):
    if not (run / "inner_complete.json").exists():
        raise ValueError("all inner fits must complete before selection")
    selections = {"source_lock_sha256": lock_hash, "rule": "inner person-balanced DeltaE00 only; checkpoint and LR jointly selected", "roles": {}}
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        idx = np.flatnonzero(fit)
        selections["roles"][role] = {}
        for family in FAMILIES:
            for fold in range(3):
                if not completed_bank(run / "inner" / role / family / f"fold{fold}", lock_hash):
                    raise ValueError("incomplete inner bank")
            candidates = []
            for checkpoint in CHECKPOINTS:
                outputs, _ = read_oof(run, role, family, checkpoint, idx)
                for lr in LRS:
                    slot_idx = [i for i, (_, rate) in enumerate(SLOTS) if rate == lr]
                    record, _ = seed_averaged_policy(outputs[slot_idx], 0., idx, data)
                    candidates.append({"lr": lr, "checkpoint": checkpoint, **record})
            best = min(candidates, key=lambda r: (r["person_mean"], r["checkpoint"], r["lr"]))
            outputs, _ = read_oof(run, role, family, best["checkpoint"], idx)
            slot_idx = [i for i, (_, rate) in enumerate(SLOTS) if rate == best["lr"]]
            policies = []
            for threshold in THRESHOLDS if family.startswith("recur_") else (0.,):
                record, _ = seed_averaged_policy(outputs[slot_idx], threshold, idx, data)
                policies.append({"threshold": threshold, **record})
            eligible = [r for r in policies if r["person_mean"] <= best["person_mean"] + .02 and r["p90"] <= best["p90"] + .1]
            chosen = min(eligible, key=lambda r: (r["mean_passes"], r["threshold"]))
            selections["roles"][role][family] = {"lr": best["lr"], "checkpoint": best["checkpoint"],
                "exit_threshold": chosen["threshold"], "selected_fixed": best, "selected_policy": chosen,
                "candidates": candidates, "policies": policies}
    path = run / "selections.json"
    if path.exists() and read_json(path) != selections:
        raise ValueError("previous selection differs; never silently revise after outer evaluation")
    write_json(path, selections)
    print(f"SELECTION LOCKED {sha(path)}", flush=True)


def final_stage(run, data, device, lock_hash):
    selections = read_json(run / "selections.json")
    if selections["source_lock_sha256"] != lock_hash:
        raise ValueError("selection source lock mismatch")
    selection_hash = sha(run / "selections.json")
    number = 0
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        idx = np.flatnonzero(fit)
        for family in FAMILIES:
            number += 1
            bank_name = f"final/{role}/{family}"
            directory = run / bank_name
            if completed_bank(directory, lock_hash):
                if read_json(directory / "receipt.json")["selection_sha256"] != selection_hash:
                    raise ValueError("final fit selection mismatch")
                continue
            selected = selections["roles"][role][family]
            print(f"START {number}/15 {bank_name}", flush=True)
            models, receipt = train_bank(data["color"][idx], data["tokens"][idx], data["target"][idx],
                weights_for(data["patient"][idx], data["site"][idx]), family,
                [(s, selected["lr"]) for s in SEEDS], selected["checkpoint"], (selected["checkpoint"],), device,
                progress_callback(run, bank_name, number, 15), engine="cuda_graph" if device.startswith("cuda") else "eager")
            payload = models[selected["checkpoint"]]
            files = {}
            for slot, seed in enumerate(SEEDS):
                path = directory / f"seed{seed}.npz"
                atomic_npz(path, **{**payload, "theta": payload["theta"][slot], "exit_threshold": np.asarray(selected["exit_threshold"], np.float32)})
                files[path.name] = sha(path)
            receipt.update({"source_lock_sha256": lock_hash, "selection_sha256": selection_hash, "role": role, "files": files})
            write_json(directory / "receipt.json", receipt)
    write_json(run / "final_complete.json", {"source_lock_sha256": lock_hash, "selection_sha256": selection_hash, "banks": 15, "independent_refits": 45})


def evaluate_stage(run, data, device, lock_hash):
    if not (run / "final_complete.json").exists():
        raise ValueError("all final refits must complete before outer evaluation")
    selections = read_json(run / "selections.json")
    final_receipt = read_json(run / "final_complete.json")
    if final_receipt["source_lock_sha256"] != lock_hash or final_receipt["selection_sha256"] != sha(run / "selections.json"):
        raise ValueError("final completion binding mismatch")
    records = []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        idx = np.flatnonzero(held)
        anchor_recorded = False
        for family in FAMILIES:
            if not completed_bank(run / "final" / role / family, lock_hash):
                raise ValueError("incomplete final bank")
            selected = selections["roles"][role][family]
            for seed in SEEDS:
                model_path = run / "final" / role / family / f"seed{seed}.npz"
                with np.load(model_path, allow_pickle=False) as z:
                    payload = dict(z)
                theta = payload["theta"]
                predictions, connections = predict_bank({**payload, "theta": theta[None]}, data["color"][idx], data["tokens"][idx], device)
                prediction, steps = apply_exit_policy(predictions[0], selected["exit_threshold"])
                record = {"role": role, "family": family, "seed": seed, "lr": selected["lr"], "checkpoint": selected["checkpoint"],
                          "exit_threshold": selected["exit_threshold"], "model_sha256": sha(model_path),
                          "learned_parameters": theta.size,
                          "numeric_payload_bytes": sum(a.nbytes for a in payload.values() if np.issubdtype(a.dtype, np.number)),
                          "archive_bytes": model_path.stat().st_size,
                          "fixed_metrics": metrics(predictions[0, :, -1], data["target"][idx], data["patient"][idx], data["site"][idx], data["device"][idx]),
                          "adaptive_metrics": metrics(prediction, data["target"][idx], data["patient"][idx], data["site"][idx], data["device"][idx]),
                          "mean_passes": float(steps.mean()), "pass_counts": {str(k): int((steps == k).sum()) for k in np.unique(steps)},
                          "mean_connections_full_trace": float(connections.mean()),
                          "connections_min": float(connections.min()), "connections_max": float(connections.max()),
                          "pass_metrics": [metrics(p, data["target"][idx], data["patient"][idx], data["site"][idx]) for p in predictions[0].transpose(1, 0, 2)]}
                records.append(record)
                atomic_npz(run / "evaluated" / role / family / f"seed{seed}.npz", row_indices=idx, predictions=predictions[0], adaptive_prediction=prediction, steps=steps, connections=connections[0])
                if not anchor_recorded:
                    _, _, anchor = transform(payload, data["color"][idx], data["tokens"][idx])
                    anchor = anchor * payload["y_std"] + payload["y_mean"]
                    records.append({"role": role, "family": "ridge_anchor", "seed": None,
                                    "fixed_metrics": metrics(anchor, data["target"][idx], data["patient"][idx], data["site"][idx], data["device"][idx])})
                    atomic_npz(run / "evaluated" / role / "ridge_anchor.npz", row_indices=idx, predictions=anchor)
                    anchor_recorded = True
        print(f"EVALUATED {role}", flush=True)
    write_json(run / "results.json", {"source_lock_sha256": lock_hash, "selection_sha256": sha(run / "selections.json"),
        "evidence": "exploratory, three overlapping historically exposed TRAIN-only protocols", "records": records})
    write_json(run / "progress.json", {"status": "fit_and_evaluation_complete_audit_pending", "pid": os.getpid(), "updated_unix": time.time()})


def smoke(device, steps, engine):
    rng = np.random.default_rng(23013)
    x = rng.normal(size=(192, 36)).astype(np.float32)
    patches = rng.normal(size=(192, 64, 18)).astype(np.float32)
    y = (np.column_stack((50 + x[:, 0] ** 2 * 5, 4 + x[:, 1] * x[:, 2] * 4, 10 + x[:, 3] * 5))).astype(np.float32)
    results = []
    for family in FAMILIES:
        models, receipt = train_bank(x, patches, y, np.ones(len(x)), family, SLOTS, steps, (steps,), device, engine=engine)
        output, _ = predict_bank(models[steps], x[:8], patches[:8], device)
        if not np.isfinite(output).all():
            raise FloatingPointError("nonfinite synthetic smoke prediction")
        results.append({"family": family, "steps": steps, "engine": engine, "bank_seconds": receipt["bank_total_seconds"],
                        "parameters": receipt["learned_parameters_per_model"], "peak_bytes": receipt["cuda_peak_allocated_bytes"]})
        print(json.dumps(results[-1]), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "inner", "select", "final", "evaluate", "run"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-steps", type=int, default=128)
    parser.add_argument("--smoke-engine", choices=("eager", "cuda_graph"), default="cuda_graph")
    args = parser.parse_args()
    setup(args.device)
    if args.stage == "smoke":
        smoke(args.device, args.smoke_steps, args.smoke_engine)
        return
    if args.cache is None or args.run is None:
        parser.error("--cache and --run required for real-data stages")
    args.run.mkdir(parents=True, exist_ok=True)
    lock_hash = source_lock(args.run, args.cache, args.device)
    data = load_data(args.cache)
    if args.stage in ("inner", "run"):
        inner_stage(args.run, data, args.device, lock_hash)
    if args.stage in ("select", "run"):
        select_stage(args.run, data, lock_hash)
    if args.stage in ("final", "run"):
        final_stage(args.run, data, args.device, lock_hash)
    if args.stage in ("evaluate", "run"):
        evaluate_stage(args.run, data, args.device, lock_hash)


if __name__ == "__main__":
    main()
