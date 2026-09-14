"""Locked equivalence and paired runtime follow-up; no new model selection."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from chromaseed_condensed_exact import fit_condensed
from chromaseed_fast_kernel import fit_one, get_model, model_id
from chromaseed_fast_kernel_runtime import allocation_peak
from chromaseed_kernel import predict_kernel
from chromaseed_kernel_runtime import time_queries
from chromaseed_kernel_train import atomic_npz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def freeze(run, source, cache, output):
    parent = js(source / "source_lock.json")
    sources = dict(parent["sources"])
    for path in (Path(__file__), ROOT / "scripts/chromaseed_condensed_exact.py", ROOT / "docs/research/chromaseed_condensed_exact_v1_protocol.md", ROOT / "pyproject.toml", ROOT / "uv.lock",
                 ROOT / "scripts/chromaseed_fast_kernel_runtime.py", ROOT / "scripts/chromaseed_kernel_runtime.py", ROOT / "scripts/chromaseed_refine_audit.py"):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for path, expected in parent["sources"].items():
        assert sha(ROOT / path) == expected
    assert sha(cache) == CACHE_HASH
    audit = js(output / "audit.json")
    assert audit["passed"] and audit["source_lock_sha256"] == sha(source / "source_lock.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_fast_kernel_audit.py")
    lock = {"sources": sources, "cache_sha256": CACHE_HASH, "source_run_lock_sha256": sha(source / "source_lock.json"),
            "source_selections_sha256": sha(source / "selections.json"), "source_results_sha256": sha(source / "results.json"),
            "parent_audit_sha256": sha(output / "audit.json"), "parent_runtime_sha256": sha(output / "runtime.json"),
            "scipy": scipy.__version__, "numpy": np.__version__, "threads": 1,
            "purpose": "equivalence/performance only; no new hyperparameter selection"}
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == lock
    else:
        write_json(path, lock)
    return sha(path)


def execute(run, source, cache, output):
    started = time.perf_counter()
    run.mkdir(parents=True, exist_ok=True)
    lock_hash = freeze(run, source, cache, output)
    choices = js(source / "selections.json")
    source_records = {(r["role"], r["arm"], r["rank"], r["seed"]): r for r in js(source / "results.json")["records"]}
    with np.load(cache, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("color", "target", "patient", "site", "device")}
    records, timing, standalone, scaling = [], [], [], []
    role_masks = roles(data["patient"], data["device"])
    for final in (False, True):
        for role, (fit, held) in role_masks.items():
            fit_rows = np.flatnonzero(fit)
            folds = folds_for(data["patient"][fit], data["device"][fit])
            for fold in ((None,) if final else (0, 1, 2)):
                indices = fit_rows if final else fit_rows[folds != fold]
                queries = np.flatnonzero(held) if final else fit_rows[folds == fold]
                assert not set(data["patient"][indices]) & set(data["patient"][queries])
                stage = "final" if final else "inner"
                bank_path = source / stage / role / ("bank" if final else f"fold{fold}")
                receipt = js(bank_path / "receipt.json")
                for name, expected in receipt["files"].items():
                    assert sha(bank_path / name) == expected
                bank = nz(bank_path / "models.npz")
                reference_oof = None if final else nz(bank_path / "oof.npz")
                weights = weights_for(data["patient"][indices], data["site"][indices])
                write_json(run / "progress.json", {"status": "reproducing", "stage": stage, "role": role, "fold": fold, "pid": os.getpid(), "updated_unix": time.time()})
                for rank in (64, 128, 256):
                    chosen = choices["roles"][role]["primary"]["column_exact"][str(rank)]
                    wi, ai = chosen["width_index"], chosen["alpha_index"]
                    for seed in (17, 29, 43):
                        key = model_id("column_exact", rank, seed, wi, ai)
                        reference = get_model(bank, key)
                        config = receipt["models"][key]
                        model, info = fit_condensed(data["color"][indices], data["target"][indices], weights, rank, seed, wi, ai)
                        for field in ("x_mean", "x_std", "y_mean", "y_std"):
                            np.testing.assert_array_equal(model[field], reference[field])
                        relative_width = abs(info["base_width"] - config["base_width"]) / config["base_width"]
                        ulps = abs(int(model["width"].view(np.uint32)) - int(reference["width"].view(np.uint32)))
                        assert relative_width <= 1e-12 and ulps <= 1
                        prediction = predict_kernel(model, data["color"][queries])
                        if final:
                            saved_path = source / "evaluated" / role / f"column_exact_k{rank:03d}_s{seed}.npz"
                            assert sha(saved_path) == source_records[(role, "column_exact", rank, seed)]["prediction_sha256"]
                            saved = nz(saved_path)
                        else:
                            saved = {"row_indices": reference_oof["row_indices"], "prediction": reference_oof[f"pred__{key}"]}
                        np.testing.assert_array_equal(saved["row_indices"], queries)
                        drift = float(np.max(np.abs(prediction - saved["prediction"])))
                        assert drift <= .002, (role, fold, rank, seed, drift)
                        name = f"condensed_exact_k{rank:03d}_s{seed}.npz"
                        path = run / "selected" / role / name if final else run / "inner" / role / f"fold{fold}" / name
                        atomic_npz(path, model)
                        record = {"stage": stage, "role": role, "fold": fold, "rank": rank, "seed": seed, "width_index": wi, "alpha_index": ai,
                                  "relative_width_difference": relative_width, "rounded_width_ulps": ulps, "max_prediction_component_drift": drift,
                                  "checked_query_rows": len(queries), "centers_exactly_equal": bool(np.array_equal(model["centers"], reference["centers"])),
                                  "payload_exactly_equal": all(np.array_equal(model[field], reference[field]) for field in model),
                                  "numeric_bytes": sum(v.nbytes for v in model.values()), "model_sha256": sha(path)}
                        if final:
                            record["metrics"] = error_summary(prediction, data["target"][queries], data["patient"][queries], data["site"][queries])[0]
                            atomic_npz(run / "evaluated" / role / name, {"row_indices": queries, "prediction": prediction})
                            measured = time_queries(model, "condensed_exact", data["color"][queries], prediction)
                            timing.append({"role": role, "rank": rank, "seed": seed, **measured})
                        records.append(record)
                print(f"REPRODUCED {stage}/{role}/fold={fold}", flush=True)
    assert len(records) == 108 and sum(r["stage"] == "final" for r in records) == 27
    write_json(run / "reproduction.json", {"source_lock_sha256": lock_hash, "records": records})
    for role, (fit, _) in role_masks.items():
        x, y, person, site = (data[k][fit] for k in ("color", "target", "patient", "site"))
        for rank in (64, 128, 256):
            chosen = choices["roles"][role]["primary"]["column_exact"][str(rank)]
            for implementation in ("dense_exact", "column_exact", "condensed_exact"):
                samples = []
                for repetition in range(4):
                    before = time.perf_counter()
                    weights = weights_for(person, site)
                    args = (rank, 17, chosen["width_index"], chosen["alpha_index"])
                    if implementation == "condensed_exact":
                        model, info = fit_condensed(x, y, weights, *args)
                    else:
                        model, info = fit_one(x, y, weights, implementation, *args)
                    elapsed = time.perf_counter() - before
                    if repetition:
                        samples.append(elapsed)
                standalone.append({"role": role, "rank": rank, "implementation": implementation, "seed": 17,
                                   "n_fit_rows": len(x), "width_index": chosen["width_index"], "alpha_index": chosen["alpha_index"],
                                   "seconds": samples, "median_ms": float(np.median(samples) * 1000)})
        print(f"PAIRED TIMING {role}", flush=True)
    rng = np.random.default_rng(829710)
    full_x = rng.normal(size=(8192, 36)).astype(np.float32)
    full_y = rng.normal([50, 4, 10], [12, 6, 9], size=(8192, 3))
    person, site = np.arange(8192) // 32, np.arange(8192) // 4
    for n in (1024, 4096, 8192):
        print(f"SCALING condensed exact N={n}", flush=True)

        def call():
            weights = weights_for(person[:n], site[:n])
            return fit_condensed(full_x[:n], full_y[:n], weights, 128, 17, 1, 1)

        samples = []
        for repetition in range(4):
            before = time.perf_counter()
            model, info = call()
            elapsed = time.perf_counter() - before
            if repetition:
                samples.append(elapsed)
        peak, model, info = allocation_peak(call)
        scaling.append({"n_rows": n, "rank": 128, "implementation": "condensed_exact", "seconds": samples,
                        "median_ms": float(np.median(samples) * 1000), "incremental_tracemalloc_peak_bytes": peak,
                        "distance_vector_bytes": info["width_info"]["condensed_distance_vector_bytes"]})
    assert freeze(run, source, cache, output) == lock_hash
    result = {"passed": True, "source_lock_sha256": lock_hash, "source_run_lock_sha256": sha(source / "source_lock.json"),
              "source_selections_sha256": sha(source / "selections.json"), "parent_audit_sha256": sha(output / "audit.json"),
              "scipy": scipy.__version__, "records": records, "inference": timing, "standalone_fit_records": standalone, "synthetic_scaling": scaling,
              "checks": {"inner_model_reproductions": 81, "final_model_reproductions": 27, "query_rows": sum(r["checked_query_rows"] for r in records),
                         "exact_payload_matches": sum(r["payload_exactly_equal"] for r in records), "exact_center_matches": sum(r["centers_exactly_equal"] for r in records)},
              "maxima": {"relative_width_difference": max(r["relative_width_difference"] for r in records),
                         "rounded_width_ulps": max(r["rounded_width_ulps"] for r in records),
                         "prediction_component_drift": max(r["max_prediction_component_drift"] for r in records)},
              "measurement": {"warmups": 1, "fit_repetitions": 3, "paired_standalone_fits_including_warmups": 108,
                              "synthetic_fits_including_allocation_trace": 15, "threads": 1, "cpu": "AMD Ryzen 9 7900X",
                              "scope": "Full independent fixed-parameter fit including balancing, normalization, exact bandwidth, pivots and solve. No cached width/matrix; excludes loading/selection/persistence. Inference includes normalization, excludes image/face preprocessing. Tracemalloc is incremental tracked allocation, not process RSS; synthetic controls in primary KF runtime were collected separately on this same host."},
              "main_total_seconds_including_reproduction_and_profiling": time.perf_counter() - started,
              "evidence": "implementation equivalence to independently audited exact KF models on reused exploratory rows; no new accuracy selection or fresh validation"}
    write_json(run / "results.json", result)
    write_json(output / "condensed_exact.json", result)
    write_json(run / "progress.json", {"status": "reproduction_and_timing_complete", "pid": os.getpid(), "updated_unix": time.time()})
    print(json.dumps({"passed": True, "checks": result["checks"], "maxima": result["maxima"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "source-run", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    execute(args.run, args.source_run, args.cache, args.output)


if __name__ == "__main__":
    main()
