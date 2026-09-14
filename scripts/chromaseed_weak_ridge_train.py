"""Expanded-ridge banks with complete exact-P control preservation."""
from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from chromaseed_fast_kernel import flatten_bank, get_model
from chromaseed_fast_kernel_train import average_metrics, js, nz, row_hash, valid_bank
from chromaseed_kernel import coordinates, gaussian_kernel, predict_kernel
from chromaseed_kernel_train import atomic_npz
from chromaseed_weak_ridge import ALPHAS, CHECKPOINTS, FAMILIES, RANK, SEEDS, WIDTHS, fit_bank, key
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


def candidate_grid(family):
    steps = (0, *CHECKPOINTS) if family.endswith("irls") else ((0,) if family == "norm_mse" else (1,))
    for wi in range(len(WIDTHS)):
        for ai in range(len(ALPHAS)):
            for step in steps:
                yield wi, ai, step


def choose_candidate(candidates):
    return min(candidates, key=lambda c: (c["person_mean"], c["steps"], c["alpha"], c["width_factor"]))


def lock_sources(run, cache, parent, exact):
    parent_lock = js(parent / "source_lock.json")
    sources = dict(parent_lock["sources"])
    for rel, expected in sources.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"P source changed: {rel}")
    if sha(exact / "source_lock.json") != parent_lock["exact_source_lock_sha256"]:
        raise ValueError("KE source binding changed")
    for rel in ("scripts/chromaseed_weak_ridge.py", "scripts/chromaseed_weak_ridge_train.py",
                "docs/research/chromaseed_weak_ridge_v1_protocol.md", "tests/test_chromaseed_weak_ridge.py",
                "scripts/chromaseed_perceptual_audit.py", "scripts/chromaseed_kernel_audit.py", "scripts/chromaseed_refine_audit.py"):
        sources[rel] = sha(ROOT / rel)
    if sha(cache) != CACHE_HASH:
        raise ValueError("only original TRAIN allowed")
    paths = [parent / "source_lock.json", parent / "selections.json", parent / "results.json"]
    for directory in sorted((parent / "inner").glob("*/fold*")) + sorted((parent / "final").glob("*/bank")):
        paths.extend([directory / "receipt.json", directory / "models.npz"])
    parent_output = ROOT / "docs/benchmarks/chromaseed_perceptual_v1"
    parent_audit = js(parent_output / "audit.json")
    if not parent_audit["passed"] or parent_audit["audit_source_sha256"] != sources["scripts/chromaseed_perceptual_audit.py"]:
        raise ValueError("P audit changed")
    lock = {"sources": sources, "cache_sha256": CACHE_HASH,
            "parent_bindings": {p.relative_to(parent).as_posix(): sha(p) for p in paths},
            "parent_audit_sha256": sha(parent_output / "audit.json"),
            "parent_verification_sha256": sha(parent_output / "verification.json"),
            "exact_source_lock_sha256": sha(exact / "source_lock.json"),
            "families": list(FAMILIES), "rank": RANK, "seeds": list(SEEDS), "alphas": list(ALPHAS),
            "widths": list(WIDTHS), "checkpoints": list(CHECKPOINTS), "tau": 2.,
            "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": platform.python_version(), "threads": 1, "change": "expanded ridge grid only"}
    path = run / "source_lock.json"
    if path.exists() and js(path) != lock:
        raise ValueError("W source lock changed")
    if not path.exists():
        write_json(path, lock)
    return sha(path)


def control(models, directory, x, rows):
    receipt = js(directory / "receipt.json")
    if receipt["fit_rows_sha256"] != row_hash(rows) or receipt["files"]["models.npz"] != sha(directory / "models.npz"):
        raise ValueError("parent bank rows or hash changed")
    arrays = nz(directory / "models.npz")
    checked = 0
    for name, config in receipt["models"].items():
        old_alpha = (.1, 1., 10.)[config["alpha_index"]]
        new_name = key(config["family"], config["seed"], config["width_index"], ALPHAS.index(old_alpha), config["steps"])
        current, previous = models[new_name], get_model(arrays, name)
        for field in previous:
            np.testing.assert_array_equal(current[field], previous[field], err_msg=f"P control changed: {name}/{field}")
        checked += 1
    if checked != 243:
        raise ValueError("all P controls required")
    return {"checked_models": checked, "identical_payloads": checked, "max_component_drift": 0.}


def predict_bank(models, x, rows):
    result = {"row_indices": rows}
    for seed in SEEDS:
        for wi in range(len(WIDTHS)):
            reference = models[key("norm_mse", seed, wi, 0, 0)]
            kernel = gaussian_kernel(coordinates(reference, x), reference["centers"], float(reference["width"]))
            for name, model in models.items():
                if f"_s{seed}_w{wi}_" not in name:
                    continue
                np.testing.assert_array_equal(model["centers"], reference["centers"])
                result[f"pred__{name}"] = kernel @ model["coefficient"].astype(np.float64) * model["y_std"] + model["y_mean"]
    return result


def run_bank(run, data, parent, role, fit, queries, fold, lock):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    directory = run / stage / role / sub
    if valid_bank(directory, lock):
        print(f"REUSE {stage}/{role}/{sub}", flush=True)
        return
    if queries is not None and set(data["patient"][fit]) & set(data["patient"][queries]):
        raise ValueError("person leakage")
    started = time.perf_counter()
    write_json(run / "progress.json", {"status": "fitting", "stage": stage, "role": role, "fold": fold, "pid": os.getpid(), "updated_unix": time.time()})
    models, receipt = fit_bank(data["color"][fit], data["target"][fit], weights_for(data["patient"][fit], data["site"][fit]))
    assert len(models) == 729
    receipt["parent_control"] = control(models, parent / stage / role / sub, data["color"][fit], fit)
    atomic_npz(directory / "models.npz", flatten_bank(models))
    files = {"models.npz": sha(directory / "models.npz")}
    if queries is not None:
        atomic_npz(directory / "oof.npz", predict_bank(models, data["color"][queries], queries))
        files["oof.npz"] = sha(directory / "oof.npz")
    receipt.update(source_lock_sha256=lock, role=role, fold=fold, fit_rows_sha256=row_hash(fit),
                   query_rows_sha256=row_hash(queries) if queries is not None else None, n_fit_people=len(np.unique(data["patient"][fit])), files=files,
                   bank_wall_seconds=time.perf_counter() - started)
    write_json(directory / "receipt.json", receipt)
    print(f"FIT {stage}/{role}/{sub}:729 readouts, {receipt['fit_bank_seconds']:.2f}s; baseline drift {receipt['parent_control']['max_component_drift']:.3g}", flush=True)


def fit_stage(run, data, parent, lock, final=False):
    if final and js(run / "selections.json")["source_lock_sha256"] != lock:
        raise ValueError("selection binding mismatch")
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        if final:
            run_bank(run, data, parent, role, rows, None, None, lock)
        else:
            folds = folds_for(data["patient"][fit], data["device"][fit])
            for fold in range(3):
                run_bank(run, data, parent, role, rows[folds != fold], rows[folds == fold], fold, lock)
    write_json(run / ("final_complete.json" if final else "inner_complete.json"),
               {"source_lock_sha256": lock, "banks": 3 if final else 9, "readouts": 2187 if final else 6561,
                "selection_sha256": sha(run / "selections.json") if final else None})


def select_stage(run, data, lock):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == lock
    selection = {"source_lock_sha256": lock, "roles": {}}
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        banks = []
        for fold in range(3):
            directory = run / "inner" / role / f"fold{fold}"
            if not valid_bank(directory, lock):
                raise ValueError("missing inner bank")
            banks.append(nz(directory / "oof.npz"))
        rows = np.concatenate([b["row_indices"] for b in banks])
        order = np.argsort(rows)
        rows = rows[order]
        np.testing.assert_array_equal(rows, np.flatnonzero(fit))
        oof = {name: np.concatenate([b[name] for b in banks])[order] for name in banks[0] if name != "row_indices"}
        choices = {}
        for family in FAMILIES:
            candidates = []
            for wi, ai, step in candidate_grid(family):
                predictions = [oof[f"pred__{key(family, seed, wi, ai, step)}"] for seed in SEEDS]
                candidates.append({"family": family, "width_index": wi, "alpha_index": ai, "steps": step,
                                   "width_factor": WIDTHS[wi], "alpha": ALPHAS[ai], **average_metrics(predictions, data, rows)})
            choices[family] = {"selected": choose_candidate(candidates), "candidates": candidates}
        selection["roles"][role] = choices
    path = run / "selections.json"
    if path.exists() and js(path) != selection:
        raise ValueError("frozen selection differs")
    write_json(path, selection)
    print(f"LOCKED 15 family choices {sha(path)}", flush=True)


def evaluate_stage(run, data, lock):
    marker = js(run / "final_complete.json")
    assert marker["source_lock_sha256"] == lock and marker["selection_sha256"] == sha(run / "selections.json")
    for role in roles(data["patient"], data["device"]):
        if not valid_bank(run / "final" / role / "bank", lock):
            raise ValueError("all final banks required before outer scoring")
    selections = js(run / "selections.json")
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        arrays = nz(run / "final" / role / "bank/models.npz")
        for family, choice in selections["roles"][role].items():
            c = choice["selected"]
            for seed in SEEDS:
                name = key(family, seed, c["width_index"], c["alpha_index"], c["steps"])
                model = get_model(arrays, name)
                prediction = predict_kernel(model, data["color"][rows])
                filename = f"{family}_s{seed}.npz"
                mp, pp = run / "selected" / role / filename, run / "evaluated" / role / filename
                atomic_npz(mp, model)
                atomic_npz(pp, {"row_indices": rows, "prediction": prediction})
                records.append({"role": role, "family": family, "seed": seed, "width_index": c["width_index"], "alpha_index": c["alpha_index"], "steps": c["steps"],
                                "numeric_bytes": sum(v.nbytes for v in model.values()), "archive_bytes": mp.stat().st_size, "model_sha256": sha(mp), "prediction_sha256": sha(pp),
                                "metrics": metrics(prediction, data["target"][rows], data["patient"][rows], data["site"][rows])})
        print(f"EVALUATED {role}", flush=True)
    assert len(records) == 45
    write_json(run / "results.json", {"source_lock_sha256": lock, "selection_sha256": sha(run / "selections.json"),
                                      "evidence": "historically reused overlapping TRAIN roles", "records": records})
    write_json(run / "progress.json", {"status": "fit_evaluation_complete_audit_pending", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inner", "select", "final", "evaluate", "run"))
    for name in ("cache", "run", "parent-run", "exact-run"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.run.mkdir(parents=True, exist_ok=True)
    resumed = (args.run / "progress.json").exists()
    lock = lock_sources(args.run, args.cache, args.parent_run, args.exact_run)
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    if args.stage in ("inner", "run"):
        fit_stage(args.run, data, args.parent_run, lock)
    if args.stage in ("select", "run"):
        select_stage(args.run, data, lock)
    if args.stage in ("final", "run"):
        fit_stage(args.run, data, args.parent_run, lock, True)
    if args.stage in ("evaluate", "run"):
        evaluate_stage(args.run, data, lock)
    if args.stage == "run":
        write_json(args.run / "workflow.json", {"source_lock_sha256": lock, "selection_sha256": sha(args.run / "selections.json"),
                                               "main_wall_seconds": time.perf_counter() - started, "resumed": resumed,
                                               "scope": "main validation, all fits, selection/evaluation/persistence; excludes interpreter/import startup, audit and profile"})


if __name__ == "__main__":
    main()


