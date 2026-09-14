"""Standalone timing after audit; no new quality selection or outer scores."""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_runtime import time_queries
from chromaseed_perceptual import fit_single
from skin_local_search_train import CACHE_HASH, roles, sha, weights_for, write_json

ROOT = Path(__file__).resolve().parents[1]


def timed_fit(data, rows, config, expected=None):
    durations, details = [], []
    for repeat in range(4):
        started = time.perf_counter()
        weights = weights_for(data["patient"][rows], data["site"][rows])
        model, info = fit_single(data["color"][rows], data["target"][rows], weights,
                                 config["family"], 17, config["width_index"], config["alpha_index"], config["steps"])
        duration = time.perf_counter() - started
        if expected is not None:
            for field in model:
                np.testing.assert_array_equal(model[field], expected[field])
        if repeat:
            durations.append(duration)
            details.append({"executed_solves": info["executed_solves"], "internal_fit_seconds": info["fit_seconds"]})
    return {**config, "n_fit_rows": len(rows), "seed": 17, "median_seconds": float(np.median(durations)), "seconds": durations, "details": details}


def runtime(run, cache, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    audit = js(output / "audit.json")
    assert audit["passed"] and audit["source_lock_sha256"] == sha(run / "source_lock.json")
    assert audit["selection_sha256"] == sha(run / "selections.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_perceptual_audit.py")
    for path, expected in {**js(run / "source_lock.json")["sources"], **audit["dependencies"]}.items():
        assert sha(ROOT / path) == expected
    assert sha(cache) == CACHE_HASH
    with np.load(cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    partitions = roles(data["patient"], data["device"])
    records, fits, fixed = [], [], []
    for record in js(run / "results.json")["records"]:
        role, family, seed = record["role"], record["family"], record["seed"]
        filename = f"{family}_s{seed}.npz"
        model_path = run / "selected" / role / filename
        prediction_path = run / "evaluated" / role / filename
        assert sha(model_path) == record["model_sha256"] and sha(prediction_path) == record["prediction_sha256"]
        model, saved = nz(model_path), nz(prediction_path)
        measured = time_queries(model, "perceptual", data["color"][saved["row_indices"]], saved["prediction"])
        records.append({"role": role, "family": family, "seed": seed, "numeric_bytes": record["numeric_bytes"], **measured})
    for role, (fit, _) in partitions.items():
        rows = np.flatnonzero(fit)
        for record in (r for r in js(run / "results.json")["records"] if r["role"] == role and r["seed"] == 17):
            config = {k: record[k] for k in ("family", "width_index", "alpha_index", "steps")}
            expected = nz(run / "selected" / role / f"{record['family']}_s17.npz")
            fits.append({"role": role, **timed_fit(data, rows, config, expected)})
        # Fixed audit-probe configurations measure the cost of rejected longer
        # training. No predictions on outer rows or new quality selection.
        for family in ("local_irls", "midpoint_irls"):
            config = {"family": family, "width_index": 1, "alpha_index": 0, "steps": 16}
            fixed.append({"role": role, **timed_fit(data, rows, config)})
        print(f"PROFILE {role}: actual single-model fits and fixed16-step cost", flush=True)
    result = {"source_lock_sha256": sha(run / "source_lock.json"), "selection_sha256": sha(run / "selections.json"),
              "runtime_source_sha256": sha(Path(__file__)), "audit_sha256": sha(output / "audit.json"),
              "dependencies": {p: sha(ROOT / p) for p in ("scripts/chromaseed_kernel_runtime.py", "scripts/chromaseed_kernel_audit.py")},
              "hardware": {"platform": platform.platform(), "processor": platform.processor(), "threads": 1, "numpy": np.__version__, "torch": torch.__version__},
              "records": records, "standalone_fit_records": fits, "fixed_step_fit_records": fixed,
              "standalone_fits_including_warmups": 60, "fixed_step_fits_including_warmups": 24,
              "inference_scope": "batch-one frozen predictor, normalization included;20 warmups/3 passes; no image/face pipeline",
              "fit_scope": "one warmup/3 complete fits including weights, normalization, exact width, landmarks/readout; no I/O/search; every selected refit payload exactly checked",
              "fixed_step_scope": "six fixed seed17/width1/alpha0.1/16-step audit-probe cases; additional timing only, not selected models or new outer quality measurements"}
    write_json(output / "runtime.json", result)
    print("PROFILE COMPLETE:45 inference models,60 selected full fits,24 fixed-step fits including warmups", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("run", "cache", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    runtime(args.run, args.cache, args.output)
