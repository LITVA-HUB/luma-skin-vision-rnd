"""Person-disjoint KF banks, frozen policy selection and isolated evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from chromaseed_fast_kernel import (
    ALPHAS,
    ARMS,
    PAIR_BUDGETS,
    RANKS,
    SEEDS,
    WIDTHS,
    evaluate_bank,
    fit_bank,
    flatten_bank,
    get_model,
    model_id,
)
from chromaseed_kernel import predict_kernel
from chromaseed_kernel_bank import get_model as old_get_model
from chromaseed_kernel_bank import model_id as old_model_id
from chromaseed_kernel_train import atomic_npz
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/research/chromaseed_fast_kernel_v1_protocol.md"


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def choose_fast(candidates):
    reference = next(c for c in candidates if c["arm"] == "column_exact")
    for pairs in PAIR_BUDGETS:
        candidate = next(c for c in candidates if c["arm"] == f"column_pairs{pairs}")
        if candidate["person_mean"] <= reference["person_mean"] + .05 and candidate["p90"] <= reference["p90"] + .10:
            return candidate
    return reference


def choose_size(candidates):
    best = min(candidates, key=lambda c: (c["person_mean"], c["rank"]))
    eligible = [c for c in candidates if c["person_mean"] <= best["person_mean"] + .05 and c["p90"] <= best["p90"] + .10]
    return min(eligible, key=lambda c: c["rank"])


def lock_sources(run, cache, parent):
    parent_lock = js(parent / "source_lock.json")
    for path, expected in parent_lock["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"parent source mismatch: {path}")
    sources = dict(parent_lock["sources"])
    for path in (Path(__file__), ROOT / "scripts/chromaseed_fast_kernel.py", PROTOCOL):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    if sha(cache) != CACHE_HASH:
        raise ValueError("only the frozen original TRAIN is authorized")
    parent_paths = [parent / "source_lock.json", parent / "selections.json"]
    for directory in sorted((parent / "inner").glob("*/fold*")) + sorted((parent / "final").glob("*/bank")):
        parent_paths.extend([directory / "receipt.json", directory / "models.npz"])
    lock = {"sources": sources, "cache_sha256": CACHE_HASH,
            "parent_bindings": {p.relative_to(parent).as_posix(): sha(p) for p in parent_paths},
            "arms": ARMS, "ranks": RANKS, "seeds": SEEDS, "alphas": ALPHAS, "width_factors": WIDTHS,
            "pair_budgets": PAIR_BUDGETS, "backend": "cpu", "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "threads": 1, "storage": "FP32", "kernel_arithmetic": "FP64"}
    path = run / "source_lock.json"
    canonical = json.loads(json.dumps(lock))
    if path.exists() and js(path) != canonical:
        raise ValueError("frozen KF sources/configuration differ")
    if not path.exists():
        write_json(path, lock)
    return sha(path)


def valid_bank(directory, lock_hash):
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        return False
    receipt = js(receipt_path)
    if receipt["source_lock_sha256"] != lock_hash:
        raise ValueError("bank source binding changed")
    for name, expected in receipt["files"].items():
        if not (directory / name).exists() or sha(directory / name) != expected:
            raise ValueError(f"bank file changed: {directory / name}")
    return True


def old_control(models, parent_bank, x_fit, fit_rows):
    arrays = nz(parent_bank / "models.npz")
    receipt = js(parent_bank / "receipt.json")
    if receipt["fit_rows_sha256"] != row_hash(fit_rows):
        raise ValueError("parent fit-row order differs")
    if receipt["files"]["models.npz"] != sha(parent_bank / "models.npz"):
        raise ValueError("parent bank checksum mismatch")
    maximum, checked = 0., 0
    for rank in (64, 128):
        for seed in SEEDS:
            for wi in range(len(WIDTHS)):
                for ai in range(len(ALPHAS)):
                    current = models[model_id("dense_exact", rank, seed, wi, ai)]
                    previous = old_get_model(arrays, old_model_id("nys_rpchol", rank, seed, wi, ai))
                    np.testing.assert_array_equal(current["centers"], previous["centers"])
                    difference = float(np.abs(predict_kernel(current, x_fit[:7]) - predict_kernel(previous, x_fit[:7])).max())
                    if difference > .002:
                        raise ValueError("old K64/K128 reproduction failed")
                    maximum = max(maximum, difference)
                    checked += 1
    return {"checked_models": checked, "max_component_drift": maximum, "probes_per_model": min(7, len(x_fit))}


def run_bank(run, data, parent, role, fit_rows, query_rows, fold, lock_hash):
    stage = "final" if fold is None else "inner"
    subdirectory = "bank" if fold is None else f"fold{fold}"
    directory = run / stage / role / subdirectory
    if valid_bank(directory, lock_hash):
        print(f"REUSE {stage}/{role}/{subdirectory}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(run / "progress.json", {"status": "fitting", "stage": stage, "role": role, "fold": fold,
                                      "pid": os.getpid(), "updated_unix": time.time()})
    if query_rows is not None and set(data["patient"][fit_rows]) & set(data["patient"][query_rows]):
        raise ValueError("fit/query person leakage")
    weights = weights_for(data["patient"][fit_rows], data["site"][fit_rows])
    models, receipt = fit_bank(data["color"][fit_rows], data["target"][fit_rows], weights)
    if len(models) != 405:
        raise ValueError("registered readout count changed")
    receipt["old_k_reproduction"] = old_control(models, parent / stage / role / subdirectory, data["color"][fit_rows], fit_rows)
    atomic_npz(directory / "models.npz", flatten_bank(models))
    files = {"models.npz": sha(directory / "models.npz")}
    if query_rows is not None:
        # No query labels enter the bank's prediction function.
        oof = evaluate_bank(models, data["color"][query_rows], query_rows)
        atomic_npz(directory / "oof.npz", oof)
        files["oof.npz"] = sha(directory / "oof.npz")
    receipt.update({"source_lock_sha256": lock_hash, "role": role, "fold": fold,
                    "fit_rows_sha256": row_hash(fit_rows), "query_rows_sha256": row_hash(query_rows) if query_rows is not None else None,
                    "n_fit_people": len(np.unique(data["patient"][fit_rows])), "files": files,
                    "bank_wall_seconds_including_parent_reproduction_and_persistence": time.perf_counter() - started})
    write_json(directory / "receipt.json", receipt)
    print(f"FIT {stage}/{role}/{subdirectory} {len(models)} readouts, fit {receipt['fit_bank_seconds']:.3f}s, total {receipt['bank_wall_seconds_including_parent_reproduction_and_persistence']:.3f}s", flush=True)


def fit_stage(run, data, parent, lock_hash, final=False):
    if final and js(run / "selections.json")["source_lock_sha256"] != lock_hash:
        raise ValueError("selection source binding mismatch")
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        indices = np.flatnonzero(fit)
        folds = folds_for(data["patient"][fit], data["device"][fit])
        if final:
            run_bank(run, data, parent, role, indices, None, None, lock_hash)
        else:
            for fold in range(3):
                run_bank(run, data, parent, role, indices[folds != fold], indices[folds == fold], fold, lock_hash)
    write_json(run / ("final_complete.json" if final else "inner_complete.json"), {
        "source_lock_sha256": lock_hash, "banks": 3 if final else 9, "readouts": 1215 if final else 3645,
        "selection_sha256": sha(run / "selections.json") if final else None})


def average_metrics(predictions, data, rows):
    records = [metrics(p, data["target"][rows], data["patient"][rows], data["site"][rows]) for p in predictions]
    return {"person_mean": float(np.mean([r["person_mean"] for r in records])), "p90": float(np.mean([r["p90"] for r in records])), "seed_metrics": records}


def select_stage(run, data, lock_hash):
    if js(run / "inner_complete.json")["source_lock_sha256"] != lock_hash:
        raise ValueError("all inner banks required")
    selection = {"source_lock_sha256": lock_hash, "roles": {}}
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        files = []
        for fold in range(3):
            directory = run / "inner" / role / f"fold{fold}"
            if not valid_bank(directory, lock_hash):
                raise ValueError("missing inner bank")
            files.append(nz(directory / "oof.npz"))
        rows = np.concatenate([f["row_indices"] for f in files])
        order = np.argsort(rows)
        rows = rows[order]
        np.testing.assert_array_equal(rows, np.flatnonzero(fit))
        oof = {key: np.concatenate([f[key] for f in files])[order] for key in files[0] if key != "row_indices"}
        primary, fast = {}, {}
        for arm in ARMS:
            primary[arm] = {}
            for rank in RANKS:
                candidates = []
                for wi, factor in enumerate(WIDTHS):
                    for ai, alpha in enumerate(ALPHAS):
                        prediction = [oof[f"pred__{model_id(arm, rank, seed, wi, ai)}"] for seed in SEEDS]
                        candidates.append({"arm": arm, "rank": rank, "width_index": wi, "alpha_index": ai, "alpha": alpha,
                                           "width_factor": factor, **average_metrics(prediction, data, rows)})
                best = min(candidates, key=lambda c: (c["person_mean"], c["alpha"], c["width_factor"]))
                primary[arm][str(rank)] = {"width_index": best["width_index"], "alpha_index": best["alpha_index"],
                                           "selected": best, "candidates": candidates}
        for rank in RANKS:
            candidates = [primary[arm][str(rank)]["selected"] for arm in ARMS if arm != "dense_exact"]
            fast[str(rank)] = {"selected": choose_fast(candidates), "candidates": candidates}
        candidates = [fast[str(rank)]["selected"] for rank in RANKS]
        size = {"selected": choose_size(candidates), "candidates": candidates}
        selection["roles"][role] = {"primary": primary, "fast": fast, "size": size}
    path = run / "selections.json"
    if path.exists() and js(path) != selection:
        raise ValueError("previously frozen selections differ")
    write_json(path, selection)
    print(f"LOCKED 45 hyperparameter, 9 fast-arm and 3 size choices: {sha(path)}", flush=True)


def evaluate_stage(run, data, lock_hash):
    marker = js(run / "final_complete.json")
    if marker["source_lock_sha256"] != lock_hash or marker["selection_sha256"] != sha(run / "selections.json"):
        raise ValueError("final marker does not match sources/choices")
    for role in roles(data["patient"], data["device"]):
        if not valid_bank(run / "final" / role / "bank", lock_hash):
            raise ValueError("all final banks required before outer evaluation")
    selection = js(run / "selections.json")
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        bank = nz(run / "final" / role / "bank/models.npz")
        for arm, rank_choices in selection["roles"][role]["primary"].items():
            for rank, chosen in rank_choices.items():
                for seed in SEEDS:
                    key = model_id(arm, int(rank), seed, chosen["width_index"], chosen["alpha_index"])
                    model = get_model(bank, key)
                    prediction = predict_kernel(model, data["color"][rows])
                    name = f"{arm}_k{int(rank):03d}_s{seed}.npz"
                    model_path = run / "selected" / role / name
                    atomic_npz(model_path, model)
                    prediction_path = run / "evaluated" / role / name
                    atomic_npz(prediction_path, {"row_indices": rows, "prediction": prediction})
                    records.append({"role": role, "arm": arm, "rank": int(rank), "seed": seed,
                                    "actual_centers": len(model["centers"]), "numeric_bytes": sum(v.nbytes for v in model.values()),
                                    "archive_bytes": model_path.stat().st_size, "width_index": chosen["width_index"], "alpha_index": chosen["alpha_index"],
                                    "model_sha256": sha(model_path), "prediction_sha256": sha(prediction_path),
                                    "metrics": metrics(prediction, data["target"][rows], data["patient"][rows], data["site"][rows])})
        print(f"EVALUATED {role}", flush=True)
    if len(records) != 135:
        raise ValueError("selected model count changed")
    write_json(run / "results.json", {"source_lock_sha256": lock_hash, "selection_sha256": sha(run / "selections.json"),
               "evidence": "historically reused overlapping original-TRAIN-only roles", "records": records})
    write_json(run / "progress.json", {"status": "fit_evaluation_complete_audit_pending", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inner", "select", "final", "evaluate", "run"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-run", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    resumed = (args.run / "progress.json").exists()
    args.run.mkdir(parents=True, exist_ok=True)
    lock_hash = lock_sources(args.run, args.cache, args.parent_run)
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("color", "target", "patient", "site", "device")}
    if args.stage in ("inner", "run"):
        fit_stage(args.run, data, args.parent_run, lock_hash)
    if args.stage in ("select", "run"):
        select_stage(args.run, data, lock_hash)
    if args.stage in ("final", "run"):
        fit_stage(args.run, data, args.parent_run, lock_hash, final=True)
    if args.stage in ("evaluate", "run"):
        evaluate_stage(args.run, data, lock_hash)
    if args.stage == "run":
        write_json(args.run / "workflow.json", {"source_lock_sha256": lock_hash, "selection_sha256": sha(args.run / "selections.json"),
                   "main_wall_seconds": time.perf_counter() - started, "resumed": resumed,
                   "scope": "main() source/data/parent verification, all inner banks, selection, final banks, outer metrics and persistence; excludes interpreter/import startup, independent audit and subsequent profiling"})


if __name__ == "__main__":
    main()
