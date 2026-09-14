"""Frozen palette representation with an analytic, compact skin readout."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from chromaseed import X_MEAN, X_STD, Y_MEAN, Y_STD, predict
from chromaseed_train import PRETRAIN_STEPS, read_trace, verify_pretraining
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    load_model,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
BASES = ("random_basis", "clean_palette", "rendered_palette", "shuffled_palette")
SEEDS = (17, 29, 43)
ALPHAS = (.1, 1., 10.)


def fit_readout(source, x, y, weights, alpha):
    start = time.perf_counter()
    z = (np.asarray(x, dtype=np.float64) - X_MEAN) / X_STD
    pre = z @ source["hidden_w"].astype(np.float64).T + source["hidden_b"]
    hidden = pre / (1 + np.exp(-np.clip(pre, -700, 700)))
    features = np.column_stack((z, hidden))
    mean, std = features.mean(0), np.maximum(features.std(0), 1e-6)
    design = np.column_stack((np.ones(len(x)), (features - mean) / std))
    weight = np.asarray(weights, dtype=np.float64)
    target = (np.asarray(y, dtype=np.float64) - Y_MEAN) / Y_STD
    if alpha <= 0 or weight.shape != (len(x),) or np.any(weight <= 0):
        raise ValueError("Positive alpha and fit weights required")
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ (weight[:, None] * design) + penalty,
                           design.T @ (weight[:, None] * target))
    unscaled = beta[1:] / std[:, None]
    intercept = beta[0] - mean @ unscaled
    result = {k: v.copy() for k, v in source.items()}
    result["skip_w"] = unscaled[:36].T.astype(np.float32)
    result["out_w"] = unscaled[36:].T.astype(np.float32)
    result["out_b"] = intercept.astype(np.float32)
    seconds = time.perf_counter() - start
    reference = (design @ beta) * Y_STD + Y_MEAN
    deployed = predict(result, x)
    return result, {"fit_seconds": seconds, "n_rows": len(x), "alpha": alpha,
                    "numeric_bytes": sum(v.nbytes for v in result.values() if v.dtype.kind == "f"),
                    "max_lab_component_drift_vs_float64_fit": float(np.max(abs(deployed - reference)))}


def source_basis(source_run, arm, seed):
    if arm == "random_basis":
        return load_model(source_run / "pretrain" / f"initial_s{seed}.npz")
    models, _ = read_trace(source_run / "pretrain" / f"{arm}_s{seed}")
    return models[PRETRAIN_STEPS]


def lock(cache, source_run, run):
    if not cache.is_absolute() or sha(cache) != CACHE_HASH:
        raise ValueError("Explicit original TRAIN path/hash required")
    verify_pretraining(source_run)
    files = [Path(__file__), ROOT / "scripts/chromaseed.py", ROOT / "scripts/chromaseed_train.py",
             ROOT / "scripts/skin_local_search_train.py", ROOT / "src/luma_skin_vision/color.py",
             ROOT / "docs/research/chromaseed_frozen_protocol.md"]
    binding = {"cache_sha256": CACHE_HASH, "pretrain_manifest_sha256": sha(source_run / "pretrain_manifest.json"),
               "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in files},
               "alphas": list(ALPHAS), "seeds": list(SEEDS), "training_device": "CPU NumPy float64",
               "numpy": np.__version__}
    path = run / "source_lock.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != binding:
            raise ValueError("Frozen-readout source binding changed")
    else:
        write_json(path, binding)


def fit_phase(data, source_run, run):
    bases = {(arm, seed): source_basis(source_run, arm, seed) for arm in BASES for seed in SEEDS}
    choices, records, finals = {}, [], []
    for protocol, (a, b) in roles(data["patient"], data["device"]).items():
        x, y, p, s, c = (data[k][a] for k in ("color", "target", "patient", "site", "device"))
        folds = folds_for(p, c)
        folder = run / protocol
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / "roles.npz", fit=a, held=b, folds=folds)
        selected = {}
        for arm in BASES:
            scores = []
            for config, alpha in enumerate(ALPHAS):
                oof = {seed: np.empty_like(y) for seed in SEEDS}
                for seed in SEEDS:
                    for fold in range(3):
                        fit, held = folds != fold, folds == fold
                        model, receipt = fit_readout(bases[(arm, seed)], x[fit], y[fit], weights_for(p[fit], s[fit]), alpha)
                        pred = predict(model, x[held])
                        oof[seed][held] = pred
                        path = folder / "inner" / f"{arm}_c{config}_s{seed}_f{fold}.npz"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(path, **model)
                        records.append({"protocol": protocol, "arm": arm, "seed": seed, "fold": fold, **receipt,
                                        "sha256": sha(path), "inner_metrics": metrics(pred, y[held], p[held], s[held])})
                scores.append(float(np.mean([metrics(oof[seed], y, p, s)["person_mean"] for seed in SEEDS])))
            best = int(np.argmin(scores))
            selected[arm] = {"alpha": ALPHAS[best], "inner_person_mean": scores[best], "config_scores": scores}
            print(json.dumps({"stage": "frozen_inner", "protocol": protocol, "arm": arm, **selected[arm]}), flush=True)
            for seed in SEEDS:
                source = bases[(arm, seed)]
                model, receipt = fit_readout(source, x, y, weights_for(p, s), ALPHAS[best])
                for key in ("hidden_w", "hidden_b"):
                    np.testing.assert_array_equal(model[key], source[key])
                path = folder / "final" / f"{arm}_s{seed}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **model)
                finals.append({"protocol": protocol, "arm": arm, "seed": seed, **receipt,
                               "path": str(path.relative_to(run)), "sha256": sha(path),
                               "serialized_bytes": path.stat().st_size})
        choices[protocol] = selected
    write_json(run / "inner_records.json", records)
    write_json(run / "frozen_selections.json", {"choices": choices, "finals": finals,
                                                "role_hashes": {str(p.relative_to(run)): sha(p) for p in run.glob("*/roles.npz")}})
    print("All frozen-basis choices and 36 final models saved; no outer evaluation.", flush=True)


def evaluate_phase(data, run):
    path = run / "frozen_selections.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    for relative, digest in frozen["role_hashes"].items():
        if sha(run / relative) != digest:
            raise ValueError("Frozen role changed")
    for row in frozen["finals"]:
        if sha(run / row["path"]) != row["sha256"]:
            raise ValueError("Frozen model changed")
    records = []
    for row in frozen["finals"]:
        protocol = row["protocol"]
        with np.load(run / protocol / "roles.npz", allow_pickle=False) as roles_data:
            held = roles_data["held"]
        x, y, p, s, c = (data[k][held] for k in ("color", "target", "patient", "site", "device"))
        model = load_model(run / row["path"])
        pred = predict(model, x)
        records.append({**row, "metrics": metrics(pred, y, p, s, c)})
        np.savez((run / row["path"]).with_name(f"{row['arm']}_s{row['seed']}_outer.npz"),
                 prediction=pred, target=y, person=p, site=s, camera=c)
    write_json(run / "evaluation.json", {"evidence": "Secondary exploratory reused original-TRAIN cohorts",
                                          "frozen_selection_sha256": sha(path), "models": records})
    print("Frozen-basis outer evaluation completed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=ROOT / "experiments/runs/chromaseed_v1")
    parser.add_argument("--run", type=Path, default=ROOT / "experiments/runs/chromaseed_frozen_v1")
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    lock(args.cache, args.source_run, args.run)
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    start = time.perf_counter()
    if args.stage == "fit":
        fit_phase(data, args.source_run, args.run)
    else:
        evaluate_phase(data, args.run)
    write_json(args.run / f"{args.stage}_completion.json", {"seconds": time.perf_counter() - start,
                                                            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
