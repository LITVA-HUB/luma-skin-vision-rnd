"""Independent row/selection/numerical audit and CPU timing of ChromaSeed-R."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chromaseed_refine_numpy import NumpyRefiner
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json

from luma_skin_vision.color import delta_e00


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def error_summary(prediction, y, person, site):
    error = delta_e00(prediction, y)
    person_errors = np.array([error[person == p].mean() for p in np.unique(person)])
    site_errors = [np.mean([error[(person == p) & (site == s)].mean() for s in np.unique(site[person == p])])
                   for p in np.unique(person)]
    return {"person_mean": float(person_errors.mean()), "image_mean": float(error.mean()),
            "site_person_mean": float(np.mean(site_errors)), "median": float(np.median(error)),
            "p90": float(np.quantile(error, .9)), "gt5": float(np.mean(error > 5)),
            "gt10": float(np.mean(error > 10)), "n_images": len(error), "n_people": len(person_errors)}, person_errors


def independent_policy(trace, threshold):
    selected, steps = [], []
    for sample in trace:
        used = len(sample)
        if threshold > 0:
            for j in range(1, len(sample)):
                if np.sqrt(np.sum((sample[j] - sample[j - 1]) ** 2)) <= threshold:
                    used = j + 1
                    break
        selected.append(sample[used - 1])
        steps.append(used)
    return np.stack(selected), np.array(steps)


def verify_preprocessor(payload, x, patches, y, weights):
    x, patches, y = x.astype(np.float64), patches.astype(np.float64), y.astype(np.float64)
    expected = {"x_mean": x.mean(0), "x_std": np.maximum(x.std(0), 1e-6),
                "t_mean": patches.mean((0, 1)), "t_std": np.maximum(patches.std((0, 1)), 1e-6),
                "y_mean": y.mean(0), "y_std": np.maximum(y.std(0), 1e-6)}
    for key, value in expected.items():
        np.testing.assert_array_equal(payload[key], value.astype(np.float32))
    x = (x - expected["x_mean"]) / expected["x_std"]
    y = (y - expected["y_mean"]) / expected["y_std"]
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(37)
    penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty, design.T @ (weights[:, None] * y))
    np.testing.assert_allclose(payload["anchor"], beta.astype(np.float32), atol=2e-6, rtol=2e-6)


def check_metric(stored, recomputed, tolerance=1e-10):
    for key in ("person_mean", "image_mean", "site_person_mean", "median", "p90", "gt5", "gt10"):
        if abs(stored[key] - recomputed[key]) > tolerance:
            raise ValueError(f"metric {key} disagrees: {stored[key]} vs {recomputed[key]}")


def audit(run, cache, output):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("only original TRAIN cache allowed")
    lock = read_json(run / "source_lock.json")
    lock_hash = sha(run / "source_lock.json")
    for path, digest in lock["sources"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"frozen source changed: {path}")
    with np.load(cache, allow_pickle=False) as z:
        data = {key: z[key] for key in ("color", "tokens", "target", "patient", "site", "device")}
    choices = read_json(run / "selections.json")
    results = read_json(run / "results.json")
    if choices["source_lock_sha256"] != lock_hash or results["source_lock_sha256"] != lock_hash or results["selection_sha256"] != sha(run / "selections.json"):
        raise ValueError("run bindings differ")
    counters = {"artifact_hashes": 0, "normalizer_anchor_sets": 0, "inner_network_checkpoints": 0,
                "inner_prediction_probe_rows": 0, "selection_decisions": 0, "outer_models": 0, "outer_prediction_rows": 0}
    maximum_drift = 0.
    paired = []
    for role, (outer_fit, outer_held) in roles(data["patient"], data["device"]).items():
        outer_fit_idx = np.flatnonzero(outer_fit)
        outer_idx = np.flatnonzero(outer_held)
        inner_folds = folds_for(data["patient"][outer_fit], data["device"][outer_fit])
        person_error_by_family = {}
        for family in lock["families"]:
            oof = {step: [] for step in lock["checkpoints"]}
            for fold in range(3):
                directory = run / "inner" / role / family / f"fold{fold}"
                receipt = read_json(directory / "receipt.json")
                if receipt["source_lock_sha256"] != lock_hash or receipt["steps"] != 8192 or len(receipt["slots"]) != 9:
                    raise ValueError("inner bank contract mismatch")
                for name, digest in receipt["files"].items():
                    if sha(directory / name) != digest:
                        raise ValueError(f"artifact hash mismatch: {directory / name}")
                    counters["artifact_hashes"] += 1
                fit_idx = outer_fit_idx[inner_folds != fold]
                held_idx = outer_fit_idx[inner_folds == fold]
                if set(data["patient"][fit_idx]) & set(data["patient"][held_idx]):
                    raise ValueError("person leakage")
                for checkpoint in lock["checkpoints"]:
                    payload = read_npz(directory / f"bank_{checkpoint}.npz")
                    prediction_file = read_npz(directory / f"oof_{checkpoint}.npz")
                    np.testing.assert_array_equal(prediction_file["row_indices"], held_idx)
                    verify_preprocessor(payload, data["color"][fit_idx], data["tokens"][fit_idx], data["target"][fit_idx], weights_for(data["patient"][fit_idx], data["site"][fit_idx]))
                    counters["normalizer_anchor_sets"] += 1
                    probe_indices = np.unique(np.linspace(0, len(held_idx) - 1, 3).astype(int))
                    for slot in range(9):
                        model = NumpyRefiner({**payload, "theta": payload["theta"][slot]})
                        for local_row in probe_indices:
                            row = held_idx[local_row]
                            prediction, count = model.predict_trace(data["color"][row], data["tokens"][row], threshold=0.)
                            reference = prediction_file["predictions"][slot, local_row]
                            drift = float(np.abs(prediction - reference).max())
                            if drift > .002:
                                raise ValueError(f"independent inference drift {drift}: {role}/{family}/{fold}/{checkpoint}/{slot}")
                            np.testing.assert_array_equal(count, prediction_file["connections"][slot, local_row])
                            maximum_drift = max(maximum_drift, drift)
                            counters["inner_prediction_probe_rows"] += 1
                        counters["inner_network_checkpoints"] += 1
                    oof[checkpoint].append(prediction_file)
            candidates, traces = [], {}
            for checkpoint, files in oof.items():
                rows = np.concatenate([f["row_indices"] for f in files])
                order = np.argsort(rows)
                np.testing.assert_array_equal(rows[order], outer_fit_idx)
                all_outputs = np.concatenate([f["predictions"] for f in files], axis=1)[:, order]
                for rate_index, lr in enumerate(lock["lrs"]):
                    outputs = all_outputs[rate_index::3]
                    records = [error_summary(p[:, -1], data["target"][outer_fit_idx], data["patient"][outer_fit_idx], data["site"][outer_fit_idx])[0] for p in outputs]
                    candidates.append({"checkpoint": checkpoint, "lr": lr,
                        "person_mean": float(np.mean([r["person_mean"] for r in records])),
                        "p90": float(np.mean([r["p90"] for r in records]))})
                    traces[(checkpoint, lr)] = outputs
            selected = min(candidates, key=lambda r: (r["person_mean"], r["checkpoint"], r["lr"]))
            saved = choices["roles"][role][family]
            if (selected["lr"], selected["checkpoint"]) != (saved["lr"], saved["checkpoint"]):
                raise ValueError("joint selection disagrees")
            for c in candidates:
                stored = next(r for r in saved["candidates"] if r["lr"] == c["lr"] and r["checkpoint"] == c["checkpoint"])
                np.testing.assert_allclose([c["person_mean"], c["p90"]], [stored["person_mean"], stored["p90"]], rtol=0, atol=1e-10)
            policy_candidates = []
            for threshold in lock["thresholds"] if family.startswith("recur_") else [0.]:
                records, steps = [], []
                for trace in traces[(selected["checkpoint"], selected["lr"])]:
                    prediction, count = independent_policy(trace, threshold)
                    records.append(error_summary(prediction, data["target"][outer_fit_idx], data["patient"][outer_fit_idx], data["site"][outer_fit_idx])[0])
                    steps.append(count.mean())
                result = {"threshold": threshold, "person_mean": float(np.mean([r["person_mean"] for r in records])),
                          "p90": float(np.mean([r["p90"] for r in records])), "mean_passes": float(np.mean(steps))}
                policy_candidates.append(result)
            accepted = [r for r in policy_candidates if r["person_mean"] <= selected["person_mean"] + .02 and r["p90"] <= selected["p90"] + .1]
            policy = min(accepted, key=lambda r: (r["mean_passes"], r["threshold"]))
            if policy["threshold"] != saved["exit_threshold"]:
                raise ValueError("exit-policy selection disagrees")
            counters["selection_decisions"] += 1
            final_dir = run / "final" / role / family
            receipt = read_json(final_dir / "receipt.json")
            if receipt["source_lock_sha256"] != lock_hash or receipt["selection_sha256"] != sha(run / "selections.json") or receipt["steps"] != selected["checkpoint"]:
                raise ValueError("final fit binding/budget mismatch")
            for name, digest in receipt["files"].items():
                if sha(final_dir / name) != digest:
                    raise ValueError("final weight hash mismatch")
                counters["artifact_hashes"] += 1
            per_seed_errors = []
            for seed in lock["seeds"]:
                payload = read_npz(final_dir / f"seed{seed}.npz")
                verify_preprocessor(payload, data["color"][outer_fit_idx], data["tokens"][outer_fit_idx], data["target"][outer_fit_idx], weights_for(data["patient"][outer_fit_idx], data["site"][outer_fit_idx]))
                counters["normalizer_anchor_sets"] += 1
                deployed = NumpyRefiner(payload)
                evaluated = read_npz(run / "evaluated" / role / family / f"seed{seed}.npz")
                np.testing.assert_array_equal(evaluated["row_indices"], outer_idx)
                full, adaptive, executed, counts = [], [], [], []
                for row in outer_idx:
                    trace, connections = deployed.predict_trace(data["color"][row], data["tokens"][row], threshold=0.)
                    prediction, steps, _ = deployed.predict(data["color"][row], data["tokens"][row])
                    full.append(trace)
                    counts.append(connections)
                    adaptive.append(prediction)
                    executed.append(steps)
                full, adaptive = np.stack(full), np.stack(adaptive)
                drift = float(np.abs(full - evaluated["predictions"]).max())
                if drift > .002:
                    raise ValueError(f"outer independent inference drift {drift}")
                maximum_drift = max(maximum_drift, drift)
                np.testing.assert_array_equal(executed, evaluated["steps"])
                np.testing.assert_array_equal(np.stack(counts), evaluated["connections"])
                np.testing.assert_allclose(adaptive, evaluated["adaptive_prediction"], rtol=0, atol=.002)
                stored = next(r for r in results["records"] if r["role"] == role and r["family"] == family and r["seed"] == seed)
                fixed_metrics, _ = error_summary(evaluated["predictions"][:, -1], data["target"][outer_idx], data["patient"][outer_idx], data["site"][outer_idx])
                adaptive_metrics, person_errors = error_summary(evaluated["adaptive_prediction"], data["target"][outer_idx], data["patient"][outer_idx], data["site"][outer_idx])
                check_metric(stored["fixed_metrics"], fixed_metrics)
                check_metric(stored["adaptive_metrics"], adaptive_metrics)
                per_seed_errors.append(person_errors)
                counters["outer_models"] += 1
                counters["outer_prediction_rows"] += len(outer_idx)
            person_error_by_family[family] = np.mean(per_seed_errors, axis=0)
            print(f"AUDITED {role}/{family}", flush=True)
        rng = np.random.default_rng(910013)
        n_people = len(np.unique(data["patient"][outer_idx]))
        draws = rng.integers(n_people, size=(20000, n_people))
        for a, b in (("patch_mlp", "stats_mlp"), ("recur_soft", "patch_mlp"), ("recur_dynamic", "patch_mlp"), ("recur_dynamic", "recur_soft"), ("recur_dynamic", "recur_top16")):
            difference = person_error_by_family[a] - person_error_by_family[b]
            interval = np.quantile(difference[draws].mean(1), [.025, .975])
            paired.append({"role": role, "a": a, "b": b, "mean_a_minus_b": float(difference.mean()),
                           "descriptive_person_bootstrap_95": interval.tolist(), "n_people": n_people,
                           "people_a_better": int((difference < 0).sum()), "per_person_differences": difference.tolist()})
    summary = {"passed": True, "source_lock_sha256": lock_hash, "selection_sha256": sha(run / "selections.json"),
               "audit_source_sha256": sha(Path(__file__)), "numpy_source_sha256": sha(ROOT / "scripts/chromaseed_refine_numpy.py"),
               "checks": counters, "maximum_numpy_vs_cuda_native_lab_component_drift": maximum_drift,
               "paired": paired, "caveat": "Descriptive exploratory intervals on few reused people; seeds are not independent subjects and protocols overlap."}
    write_json(output / "audit.json", summary)
    return summary


def runtime(run, cache, output):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("unexpected cache")
    with np.load(cache, allow_pickle=False) as z:
        x, patches, person, device = z["color"], z["tokens"], z["patient"], z["device"]
    records = []
    choices = read_json(run / "selections.json")
    for role, (_, held) in roles(person, device).items():
        idx = np.flatnonzero(held)
        for family in choices["roles"][role]:
            for seed in (17, 29, 43):
                payload = read_npz(run / "final" / role / family / f"seed{seed}.npz")
                model = NumpyRefiner(payload)
                for policy in ("fixed", "adaptive"):
                    threshold = 0. if policy == "fixed" else choices["roles"][role][family]["exit_threshold"]
                    for row in idx[:min(20, len(idx))]:
                        model.predict(x[row], patches[row], threshold)
                    times, passes = [], []
                    for _ in range(3):
                        for row in idx:
                            started = time.perf_counter_ns()
                            _, executed, _ = model.predict(x[row], patches[row], threshold)
                            times.append((time.perf_counter_ns() - started) / 1000.)
                            passes.append(executed)
                    records.append({"role": role, "family": family, "seed": seed, "policy": policy,
                                    "median_us": float(np.median(times)), "p95_us": float(np.quantile(times, .95)),
                                    "mean_us": float(np.mean(times)), "mean_passes": float(np.mean(passes)), "n_calls": len(times)})
            print(f"TIMED {role}/{family}", flush=True)
    result = {"scope": "NumPy float32 batch-one CPU; includes normalization, anchor, gates and real loop exit; excludes image/feature extraction and face detection",
              "rows": "all held rows; 20 warmups then 3 complete repetitions per model/policy",
              "platform": platform.platform(), "processor": platform.processor(), "numpy": np.__version__,
              "source_sha256": sha(ROOT / "scripts/chromaseed_refine_numpy.py"), "records": records}
    write_json(output / "runtime.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "runtime", "all"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage in ("audit", "all"):
        audit(args.run, args.cache, args.output)
    if args.stage in ("runtime", "all"):
        runtime(args.run, args.cache, args.output)


if __name__ == "__main__":
    main()
