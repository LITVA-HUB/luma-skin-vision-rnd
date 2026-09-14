"""Verified KF inference, independent-fit timing and synthetic allocation scale."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from chromaseed_fast_kernel import fit_one
from chromaseed_kernel import predict_kernel
from chromaseed_kernel_runtime import time_queries
from skin_local_search_train import CACHE_HASH, roles, sha, weights_for, write_json

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def locks(run, output):
    for path, expected in js(run / "source_lock.json")["sources"].items():
        assert sha(ROOT / path) == expected
    audit = js(output / "audit.json")
    assert audit["passed"] and audit["selection_sha256"] == sha(run / "selections.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_fast_kernel_audit.py")
    for path, expected in audit["dependencies"].items():
        assert sha(ROOT / path) == expected


def allocation_peak(call):
    gc.collect()
    tracemalloc.start()
    before, _ = tracemalloc.get_traced_memory()
    model, info = call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak - before, model, info


def run_measurement(run, cache, output):
    locks(run, output)
    assert sha(cache) == CACHE_HASH
    with np.load(cache, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("color", "target", "patient", "site", "device")}
    records = js(run / "results.json")["records"]
    timing, fits = [], []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        for record in [r for r in records if r["role"] == role]:
            arm, rank, seed = record["arm"], record["rank"], record["seed"]
            filename = f"{arm}_k{rank:03d}_s{seed}.npz"
            path = run / "selected" / role / filename
            assert sha(path) == record["model_sha256"]
            evaluated = nz(run / "evaluated" / role / filename)
            np.testing.assert_array_equal(evaluated["row_indices"], np.flatnonzero(held))
            measured = time_queries(nz(path), arm, data["color"][held], evaluated["prediction"])
            timing.append({"role": role, "arm": arm, "rank": rank, "seed": seed, "numeric_bytes": record["numeric_bytes"], **measured})
        x, y, person, site = (data[k][fit] for k in ("color", "target", "patient", "site"))
        for record in [r for r in records if r["role"] == role and r["seed"] == 17]:
            arm, rank = record["arm"], record["rank"]
            reference = nz(run / "selected" / role / f"{arm}_k{rank:03d}_s17.npz")
            samples, drift = [], 0.
            for repetition in range(4):
                before = time.perf_counter()
                weights = weights_for(person, site)
                model, info = fit_one(x, y, weights, arm, rank, 17, record["width_index"], record["alpha_index"])
                elapsed = time.perf_counter() - before
                if repetition:
                    samples.append(elapsed)
                difference = float(np.max(np.abs(predict_kernel(model, x[:7]) - predict_kernel(reference, x[:7]))))
                assert difference <= .002
                drift = max(drift, difference)
            fits.append({"role": role, "n_fit_rows": len(x), "arm": arm, "rank": rank, "seed": 17,
                         "width_index": record["width_index"], "alpha_index": record["alpha_index"],
                         "median_ms": float(np.median(samples) * 1000), "seconds": samples, "max_refit_component_drift": drift,
                         "kernel_entries_evaluated": info["kernel_entries_evaluated"], "last_preparation_seconds": info["preparation_seconds"],
                         "landmark_phase_largest_matrix_shape": info["largest_training_matrix_shape"]})
        print(f"TIMED real {role}", flush=True)
    write_json(output / "runtime_real_checkpoint.json", {"timing": timing, "fits": fits})
    rng = np.random.default_rng(829710)
    full_x = rng.normal(size=(8192, 36)).astype(np.float32)
    full_y = rng.normal([50, 4, 10], [12, 6, 9], size=(8192, 3))
    full_person = np.arange(8192) // 32
    full_site = np.arange(8192) // 4
    scaling = []
    for n in (1024, 4096, 8192):
        x, y = full_x[:n], full_y[:n]
        for arm in ("dense_exact", "column_exact", "column_pairs4096"):
            print(f"SCALING synthetic N={n} {arm}", flush=True)

            def call():
                weights = weights_for(full_person[:n], full_site[:n])
                return fit_one(x, y, weights, arm, 128, 17, 1, 1)

            samples = []
            for repetition in range(4):
                before = time.perf_counter()
                model, info = call()
                elapsed = time.perf_counter() - before
                if repetition:
                    samples.append(elapsed)
            peak, model, info = allocation_peak(call)
            scaling.append({"n_rows": n, "arm": arm, "rank": 128, "seconds": samples, "median_ms": float(np.median(samples) * 1000),
                            "incremental_tracemalloc_peak_bytes": peak, "kernel_entries_evaluated": info["kernel_entries_evaluated"],
                            "landmark_phase_largest_matrix_shape": info["largest_training_matrix_shape"],
                            "numeric_payload_bytes": sum(v.nbytes for v in model.values())})
            write_json(output / "runtime_scaling_checkpoint.json", {"records": scaling})
    locks(run, output)
    write_json(output / "runtime.json", {"source_lock_sha256": sha(run / "source_lock.json"), "selection_sha256": sha(run / "selections.json"),
               "runtime_source_sha256": sha(Path(__file__)), "audit_sha256": sha(output / "audit.json"),
               "dependencies": {"scripts/chromaseed_kernel_runtime.py": sha(ROOT / "scripts/chromaseed_kernel_runtime.py")},
               "hardware": {"cpu": "AMD Ryzen 9 7900X", "os": platform.platform(), "threads": 1},
               "inference_scope": "Batch1 warm CPU: normalization + FP64 kernels/readouts. Same frozen K predictor. No file load, face/image feature extraction, network or app time. 20 warmups +3 full role query passes per selected model.",
               "fit_scope": "Seed17 per role/arm/rank; 1 warmup +3 independent timed fits. Includes person/site weights, normalization, bandwidth, columns and solve. No reuse between fits. Excludes loading, hyperparameter selection and persistence. Matrix-shape metadata refers to landmark stage; column_exact still allocates quadratic distances during exact bandwidth.",
               "synthetic_scope": "Synthetic Gaussian color36 and Lab targets, no image realism/accuracy evidence. Each N/arm:1 warmup+3 untraced timed fits, then1 separate tracemalloc fit. NumPy allocator visibility confirmed by preflight. Peak is incremental traced allocation above already-loaded data/runtime, not process RSS. Exact bandwidth allocation is included. No million-row scaling claim.",
               "records": timing, "standalone_fit_records": fits, "synthetic_scaling": scaling,
               "standalone_fit_count_including_warmups": 4 * len(fits), "synthetic_fit_count_including_memory_probes": 5 * len(scaling)})


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_measurement(args.run, args.cache, args.output)


if __name__ == "__main__":
    main()
