"""Original-TRAIN-only kernel study; choices precede all outer evaluation."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
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

from chromaseed_kernel import AdaptiveKernel, convex_blend, predict_kernel
from chromaseed_kernel_bank import (
    ALPHAS,
    FAMILIES,
    RANKS,
    SEEDS,
    WIDTHS,
    choose_from_trace,
    evaluate_bank,
    fit_bank,
    flatten_bank,
    get_model,
    make_adaptive,
    model_id,
    seeds_for,
)
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)
from skin_local_search_train import predict as legacy_predict

TOLERANCES = (0., .1, .25, .5, 1.)
RHOS = (0., .25, .5, .75, 1.)
ROLE_NAMES = ("mixed", "slr_to_ipod", "ipod_to_slr")
BOUND_SOURCES = ("scripts/chromaseed_kernel.py", "scripts/chromaseed_kernel_bank.py", "scripts/chromaseed_kernel_train.py",
                 "scripts/skin_local_search_train.py", "scripts/skin_local_search_core.py", "src/luma_skin_vision/color.py",
                 "docs/research/chromaseed_kernel_v1_protocol.md")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


def legacy_path(legacy, role, seed, fold=None, method="guided_rbf"):
    selected = read_json(legacy / role / "selection.json")[method]
    if fold is None:
        path = legacy / role / "final" / f"{method}_s{seed}.npz"
    else:
        ci = ALPHAS.index(selected["parameter"])
        path = legacy / role / "inner" / f"{method}_c{ci}_s{seed}_f{fold}.npz"
    receipt = read_json(path.with_suffix(".json"))
    if sha(path) != receipt["artifact_sha256"]:
        raise ValueError("legacy weight artifact changed")
    return path


def source_lock(run, cache, legacy):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("only the exact original TRAIN cache is authorized")
    old_lock = read_json(legacy / "source_lock.json")
    for name, digest in old_lock["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("legacy source changed")
    legacy_bindings = {"source_lock.json": sha(legacy / "source_lock.json"), "frozen_selections.json": sha(legacy / "frozen_selections.json")}
    for role in ROLE_NAMES:
        legacy_bindings[f"{role}/selection.json"] = sha(legacy / role / "selection.json")
        legacy_bindings[f"{role}/roles.npz"] = sha(legacy / role / "roles.npz")
        for seed in SEEDS:
            for fold in (None, 0, 1, 2):
                path = legacy_path(legacy, role, seed, fold)
                legacy_bindings[str(path.relative_to(legacy))] = sha(path)
        path = legacy_path(legacy, role, 17, method="krr")
        legacy_bindings[str(path.relative_to(legacy))] = sha(path)
    lock = {"sources": {name: sha(ROOT / name) for name in BOUND_SOURCES}, "cache_sha256": CACHE_HASH,
            "legacy_bindings": legacy_bindings, "families": FAMILIES, "ranks": RANKS, "seeds": SEEDS,
            "alphas": ALPHAS, "width_factors": WIDTHS, "tolerances": TOLERANCES, "blend_weights": RHOS,
            "exact_backend": "cuda", "compact_backend": "cpu", "storage": "float32 plus float64 residual diagnostics",
            "kernel_arithmetic": "float64", "numpy": np.__version__, "torch": torch.__version__, "python": platform.python_version()}
    lock = json.loads(json.dumps(lock))
    path = run / "source_lock.json"
    if path.exists() and read_json(path) != lock:
        raise ValueError("immutable source/config binding changed; use a distinct version/run")
    write_json(path, lock)
    return sha(path)


def load_data(cache):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("wrong dataset")
    with np.load(cache, allow_pickle=False) as z:
        return {key: z[key] for key in ("color", "target", "patient", "site", "device")}


def verify_bank(directory, lock_hash):
    if not (directory / "receipt.json").exists():
        return False
    receipt = read_json(directory / "receipt.json")
    if receipt["source_lock_sha256"] != lock_hash:
        raise ValueError("bank binding changed")
    for name, digest in receipt["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("bank artifact changed")
    return True


def run_bank(run, data, role, fit_idx, held_idx, fold, lock_hash):
    stage = "final" if fold is None else "inner"
    relative = Path(stage) / role / ("bank" if fold is None else f"fold{fold}")
    directory = run / relative
    if verify_bank(directory, lock_hash):
        return
    write_json(run / "progress.json", {"status": "running", "pid": os.getpid(), "bank": str(relative), "updated_unix": time.time()})
    started = time.perf_counter()
    models, diagnostics, receipt = fit_bank(data["color"][fit_idx], data["target"][fit_idx], weights_for(data["patient"][fit_idx], data["site"][fit_idx]))
    model_path = directory / "models.npz"
    atomic_npz(model_path, flatten_bank(models, diagnostics))
    files = {model_path.name: sha(model_path)}
    if fold is not None:
        if set(data["patient"][fit_idx]) & set(data["patient"][held_idx]):
            raise ValueError("person leakage")
        predictions, violation = evaluate_bank(models, diagnostics, receipt, data["color"][held_idx], held_idx)
        prediction_path = directory / "oof.npz"
        atomic_npz(prediction_path, predictions)
        files[prediction_path.name] = sha(prediction_path)
        receipt["maximum_oof_bound_violation_native_lab"] = violation
    receipt.update({"source_lock_sha256": lock_hash, "role": role, "fold": fold, "fit_rows_sha256": row_hash(fit_idx),
                    "held_rows_sha256": row_hash(held_idx) if fold is not None else None,
                    "n_fit_people": len(np.unique(data["patient"][fit_idx])), "files": files,
                    "bank_wall_seconds_including_persistence_and_queries": time.perf_counter() - started})
    if fold is None:
        receipt["selection_sha256"] = sha(run / "selections.json")
    write_json(directory / "receipt.json", receipt)
    print(json.dumps({"bank": str(relative), "readouts": receipt["readout_configurations"], "fit_seconds": round(receipt["fit_bank_seconds"], 3), "total_seconds": round(receipt["bank_wall_seconds_including_persistence_and_queries"], 3)}), flush=True)


def fit_stage(run, data, legacy, lock_hash, final=False):
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        indices = np.flatnonzero(fit)
        folds = folds_for(data["patient"][fit], data["device"][fit])
        old_roles = read_npz(legacy / role / "roles.npz")
        np.testing.assert_array_equal(old_roles["fit"], fit)
        np.testing.assert_array_equal(old_roles["held"], held)
        np.testing.assert_array_equal(old_roles["folds"], folds)
        if final:
            selections = read_json(run / "selections.json")
            if selections["source_lock_sha256"] != lock_hash:
                raise ValueError("selection binding mismatch")
            run_bank(run, data, role, indices, None, None, lock_hash)
        else:
            for fold in range(3):
                run_bank(run, data, role, indices[folds != fold], indices[folds == fold], fold, lock_hash)
    marker = "final_complete.json" if final else "inner_complete.json"
    write_json(run / marker, {"source_lock_sha256": lock_hash, "banks": 3 if final else 9, "readouts": 1107 if final else 3321,
                             "selection_sha256": sha(run / "selections.json") if final else None})


def load_oof(run, role, expected_idx, lock_hash):
    files, receipts = [], []
    for fold in range(3):
        directory = run / "inner" / role / f"fold{fold}"
        if not verify_bank(directory, lock_hash):
            raise ValueError("all inner banks required before selection")
        files.append(read_npz(directory / "oof.npz"))
        receipts.append(read_json(directory / "receipt.json"))
    rows = np.concatenate([f["row_indices"] for f in files])
    order = np.argsort(rows)
    np.testing.assert_array_equal(rows[order], expected_idx)
    joined = {key: np.concatenate([f[key] for f in files], axis=0)[order] for key in files[0] if key != "row_indices"}
    ranks = {key: np.concatenate([np.full(len(f["row_indices"]), r["models"][key]["actual_centers"]) for f, r in zip(files, receipts, strict=True)])[order] for key in receipts[0]["models"]}
    return joined, ranks, files


def average_metrics(predictions, data, idx):
    records = [metrics(p, data["target"][idx], data["patient"][idx], data["site"][idx]) for p in predictions]
    return {"person_mean": float(np.mean([r["person_mean"] for r in records])), "p90": float(np.mean([r["p90"] for r in records])), "seed_metrics": records}


def legacy_oof(legacy, data, role, idx, files):
    result = {}
    for seed in SEEDS:
        rows, values = [], []
        for fold, f in enumerate(files):
            row = f["row_indices"]
            rows.append(row)
            model = read_npz(legacy_path(legacy, role, seed, fold))
            values.append(legacy_predict(model, data["color"][row]))
        rows = np.concatenate(rows)
        order = np.argsort(rows)
        np.testing.assert_array_equal(rows[order], idx)
        result[seed] = np.concatenate(values)[order]
    return result


def select_stage(run, data, legacy, lock_hash):
    if not (run / "inner_complete.json").exists():
        raise ValueError("all inner fits required")
    choices = {"source_lock_sha256": lock_hash, "roles": {}}
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        idx = np.flatnonzero(fit)
        oof, actual_ranks, files = load_oof(run, role, idx, lock_hash)
        selected = {"primary": {}, "adaptive": {}, "blends": {}}
        for family in FAMILIES:
            selected["primary"][family] = {}
            for rank in (0,) if family == "exact" else RANKS:
                candidates = []
                for wi, factor in enumerate(WIDTHS):
                    for ai, alpha in enumerate(ALPHAS):
                        prediction = [oof[f"pred__{model_id(family, rank, seed, wi, ai)}"] for seed in seeds_for(family)]
                        candidates.append({"width_index": wi, "width_factor": factor, "alpha_index": ai, "alpha": alpha, **average_metrics(prediction, data, idx)})
                best = min(candidates, key=lambda r: (r["person_mean"], r["alpha"], r["width_factor"]))
                selected["primary"][family][str(rank)] = {"width_index": best["width_index"], "alpha_index": best["alpha_index"], "selected": best, "candidates": candidates}
        reference = selected["primary"]["project_rpchol"]["128"]
        wi, ai = reference["width_index"], reference["alpha_index"]
        policies = []
        for tolerance in TOLERANCES:
            outputs, counts, coverage = [], [], []
            for seed in SEEDS:
                keys = [model_id("project_rpchol", rank, seed, wi, ai) for rank in RANKS]
                traces = np.stack([oof[f"pred__{key}"] for key in keys], axis=1)
                bounds = np.stack([oof[f"bound__{key}"] for key in keys], axis=1)
                ranks = np.stack([actual_ranks[key] for key in keys], axis=1)
                prediction, used, _, met = choose_from_trace(traces, bounds, ranks, tolerance)
                outputs.append(prediction)
                counts.append(used.mean())
                coverage.append(met.mean())
            policies.append({"tolerance": tolerance, "mean_centers": float(np.mean(counts)), "tolerance_met_fraction": float(np.mean(coverage)), **average_metrics(outputs, data, idx)})
        fixed = policies[0]
        acceptable = [r for r in policies if r["person_mean"] <= fixed["person_mean"] + .02 and r["p90"] <= fixed["p90"] + .1]
        best = min(acceptable, key=lambda r: (r["mean_centers"], r["tolerance"]))
        selected["adaptive"] = {"width_index": wi, "alpha_index": ai, "tolerance": best["tolerance"], "selected": best, "policies": policies}
        alternative = legacy_oof(legacy, data, role, idx, files)
        for family, rank in (("nys_rpchol", 64), ("project_rpchol", 128)):
            reference = selected["primary"][family][str(rank)]
            wi, ai = reference["width_index"], reference["alpha_index"]
            candidates = []
            for rho in RHOS:
                prediction = [convex_blend(oof[f"pred__{model_id(family, rank, seed, wi, ai)}"], alternative[seed], rho) for seed in SEEDS]
                candidates.append({"rho": rho, **average_metrics(prediction, data, idx)})
            best = min(candidates, key=lambda r: (r["person_mean"], r["rho"]))
            selected["blends"][f"{family}_{rank}"] = {"family": family, "rank": rank, "width_index": wi, "alpha_index": ai,
                                                       "rho": best["rho"], "selected": best, "candidates": candidates}
        choices["roles"][role] = selected
    path = run / "selections.json"
    if path.exists() and read_json(path) != choices:
        raise ValueError("previously frozen choices differ")
    write_json(path, choices)
    print(f"SELECTIONS LOCKED {sha(path)}", flush=True)


def numerical_bytes(payload):
    return sum(v.nbytes for v in payload.values() if np.issubdtype(v.dtype, np.number))


def evaluate_stage(run, data, legacy, lock_hash):
    marker = read_json(run / "final_complete.json")
    if marker["source_lock_sha256"] != lock_hash or marker["selection_sha256"] != sha(run / "selections.json"):
        raise ValueError("all final banks must match frozen choices")
    choices = read_json(run / "selections.json")
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        idx = np.flatnonzero(held)
        directory = run / "final" / role / "bank"
        if not verify_bank(directory, lock_hash):
            raise ValueError("missing final bank")
        bank = read_npz(directory / "models.npz")
        choice = choices["roles"][role]
        for family, rank_choices in choice["primary"].items():
            for rank, selected in rank_choices.items():
                for seed in seeds_for(family):
                    key = model_id(family, int(rank), seed, selected["width_index"], selected["alpha_index"])
                    model = get_model(bank, key)
                    prediction = predict_kernel(model, data["color"][idx])
                    model_path = run / "selected" / role / f"{family}_k{int(rank):03d}_s{seed}.npz"
                    atomic_npz(model_path, model)
                    atomic_npz(run / "evaluated" / role / f"{family}_k{int(rank):03d}_s{seed}.npz", {"row_indices": idx, "prediction": prediction})
                    records.append({"role": role, "family": family, "rank": int(rank), "actual_centers": len(model["centers"]), "seed": seed,
                                    "width_index": selected["width_index"], "alpha_index": selected["alpha_index"],
                                    "numeric_bytes": numerical_bytes(model), "archive_bytes": model_path.stat().st_size, "model_sha256": sha(model_path),
                                    "metrics": metrics(prediction, data["target"][idx], data["patient"][idx], data["site"][idx])})
        adaptive_choice = choice["adaptive"]
        for seed in SEEDS:
            model = make_adaptive(bank, adaptive_choice["width_index"], adaptive_choice["alpha_index"], seed, adaptive_choice["tolerance"])
            deployed = AdaptiveKernel(model)
            trace, bounds = deployed.full_trace(data["color"][idx])
            prediction, used, bound, met = choose_from_trace(trace, bounds, deployed.ranks, adaptive_choice["tolerance"])
            teacher = get_model(bank, model_id("exact", 0, 17, adaptive_choice["width_index"], adaptive_choice["alpha_index"]))
            teacher_prediction = predict_kernel(teacher, data["color"][idx])
            difference = np.linalg.norm(prediction - teacher_prediction, axis=1)
            per_level_difference = np.linalg.norm(trace - teacher_prediction[:, None], axis=-1)
            violation = float(np.max(per_level_difference - bounds))
            if violation > 1e-6:
                raise ValueError("adaptive teacher-bound violation")
            model_path = run / "selected" / role / f"adaptive_project_s{seed}.npz"
            atomic_npz(model_path, model)
            atomic_npz(run / "evaluated" / role / f"adaptive_project_s{seed}.npz", {"row_indices": idx, "prediction": prediction, "trace": trace, "bounds": bounds,
                        "used_centers": used, "selected_bounds": bound, "tolerance_met": met, "teacher_prediction": teacher_prediction})
            records.append({"role": role, "family": "adaptive_project", "rank": 128, "seed": seed, "tolerance": adaptive_choice["tolerance"],
                            "numeric_bytes": numerical_bytes(model), "archive_bytes": model_path.stat().st_size, "model_sha256": sha(model_path),
                            "mean_centers": float(used.mean()), "rank_counts": {str(k): int((used == k).sum()) for k in np.unique(used)},
                            "tolerance_met_fraction": float(met.mean()), "max_teacher_native_lab_distance": float(difference.max()),
                            "max_bound_violation": violation, "metrics": metrics(prediction, data["target"][idx], data["patient"][idx], data["site"][idx])})
        for name, blend in choice["blends"].items():
            for seed in SEEDS:
                base = get_model(bank, model_id(blend["family"], blend["rank"], seed, blend["width_index"], blend["alpha_index"]))
                guided = read_npz(legacy_path(legacy, role, seed))
                prediction = convex_blend(predict_kernel(base, data["color"][idx]), legacy_predict(guided, data["color"][idx]), blend["rho"])
                payload = {"rho": np.asarray(blend["rho"], np.float32)}
                if blend["rho"] < 1:
                    payload.update({f"base__{key}": value for key, value in base.items()})
                if blend["rho"] > 0:
                    payload.update({f"guided__{key}": value for key, value in guided.items()})
                model_path = run / "selected" / role / f"blend_{name}_s{seed}.npz"
                atomic_npz(model_path, payload)
                atomic_npz(run / "evaluated" / role / f"blend_{name}_s{seed}.npz", {"row_indices": idx, "prediction": prediction})
                records.append({"role": role, "family": f"blend_{name}", "rank": blend["rank"], "seed": seed, "rho": blend["rho"],
                                "numeric_bytes": numerical_bytes(payload), "archive_bytes": model_path.stat().st_size, "model_sha256": sha(model_path),
                                "metrics": metrics(prediction, data["target"][idx], data["patient"][idx], data["site"][idx])})
        # A separate width-factor-one reproduction control retains the old alpha.
        old_alpha = read_json(legacy / role / "selection.json")["krr"]["parameter"]
        reference = get_model(bank, model_id("exact", 0, 17, 1, ALPHAS.index(old_alpha)))
        prediction = predict_kernel(reference, data["color"][idx])
        previous = legacy_predict(read_npz(legacy_path(legacy, role, 17, method="krr")), data["color"][idx])
        reproduced = metrics(prediction, data["target"][idx], data["patient"][idx], data["site"][idx])
        prior_metrics = metrics(previous, data["target"][idx], data["patient"][idx], data["site"][idx])
        if abs(reproduced["person_mean"] - prior_metrics["person_mean"]) > .001:
            raise ValueError("old KRR reproduction exceeds registered numerical tolerance")
        records.append({"role": role, "family": "exact_width1_reference", "rank": 0, "seed": 17,
                        "metrics": reproduced, "legacy_person_mean": prior_metrics["person_mean"], "max_native_lab_component_drift": float(np.abs(prediction - previous).max()),
                        "numeric_bytes": numerical_bytes(reference)})
        print(f"EVALUATED {role}", flush=True)
    write_json(run / "results.json", {"source_lock_sha256": lock_hash, "selection_sha256": sha(run / "selections.json"),
               "evidence": "historically exposed overlapping original-TRAIN-only roles", "records": records})
    write_json(run / "progress.json", {"status": "fit_evaluation_complete_audit_pending", "pid": os.getpid(), "updated_unix": time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inner", "select", "final", "evaluate", "run"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, default=ROOT / "experiments/runs/skin_local_search_v1")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.run.mkdir(parents=True, exist_ok=True)
    lock_hash = source_lock(args.run, args.cache, args.legacy_run)
    data = load_data(args.cache)
    if args.stage in ("inner", "run"):
        fit_stage(args.run, data, args.legacy_run, lock_hash)
    if args.stage in ("select", "run"):
        select_stage(args.run, data, args.legacy_run, lock_hash)
    if args.stage in ("final", "run"):
        fit_stage(args.run, data, args.legacy_run, lock_hash, final=True)
    if args.stage in ("evaluate", "run"):
        evaluate_stage(args.run, data, args.legacy_run, lock_hash)


if __name__ == "__main__":
    main()
