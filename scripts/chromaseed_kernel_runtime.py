"""Actual batch-one execution and unamortized fit timing for frozen K models."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from chromaseed_kernel import (
    AdaptiveKernel,
    coordinates,
    exact_coefficients,
    fit_normalizer,
    gaussian_kernel,
    median_width,
    nystrom_coefficients,
    predict_kernel,
    projection_coefficients,
    select_landmarks,
)
from chromaseed_kernel_bank import ALPHAS, WIDTHS
from skin_local_search_train import CACHE_HASH, roles, sha, weights_for, write_json
from skin_local_search_train import predict as legacy_predict

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def check_lock(run):
    lock = js(run / "source_lock.json")
    for path, expected in lock["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"frozen source changed: {path}")


def filename(record):
    family, seed = record["family"], record["seed"]
    if family == "adaptive_project" or family.startswith("blend_"):
        return f"{family}_s{seed}.npz"
    return f"{family}_k{record['rank']:03d}_s{seed}.npz"


def predictor(payload, family):
    if family == "adaptive_project":
        adaptive = AdaptiveKernel(payload)
        return lambda x: adaptive.predict_one(x)[0]
    if family.startswith("blend_"):
        rho = float(payload["rho"])
        base = {k.removeprefix("base__"): v for k, v in payload.items() if k.startswith("base__")}
        guided = {k.removeprefix("guided__"): v for k, v in payload.items() if k.startswith("guided__")}
        if rho == 0.:
            return lambda x: predict_kernel(base, x[None])[0]
        if rho == 1.:
            return lambda x: legacy_predict(guided, x[None])[0]
        return lambda x: (1. - rho) * predict_kernel(base, x[None])[0] + rho * legacy_predict(guided, x[None])[0]
    if family.startswith("legacy_"):
        return lambda x: legacy_predict(payload, x[None])[0]
    return lambda x: predict_kernel(payload, x[None])[0]


def time_queries(payload, family, query, expected, warmups=20, repetitions=3):
    before = time.perf_counter_ns()
    call = predictor(payload, family)
    init_us = (time.perf_counter_ns() - before) / 1000
    actual = np.stack([call(x) for x in query])
    drift = float(np.abs(actual - expected).max())
    tolerance = .002 if family.startswith(("legacy_", "blend_")) else 2e-8
    if drift > tolerance:
        raise ValueError(f"batch-one prediction disagrees: {family}, {drift}")
    for i in range(warmups):
        call(query[i % len(query)])
    durations = []
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            for x in query:
                before = time.perf_counter_ns()
                call(x)
                durations.append((time.perf_counter_ns() - before) / 1000)
    finally:
        if gc_enabled:
            gc.enable()
    return {"median_us": float(np.median(durations)), "p95_us": float(np.percentile(durations, 95)),
            "mean_us": float(np.mean(durations)), "query_count": len(durations), "verification_rows": len(query),
            "max_native_lab_component_drift": drift, "init_us_excluding_file_load": init_us}


def standalone_fit(x, y, person, site, family, rank, config, exact_backend):
    weights = weights_for(person, site)
    prep = fit_normalizer(x, y)
    normalized = coordinates(prep, x)
    yn = (y.astype(np.float64) - prep["y_mean"]) / prep["y_std"]
    width = float(np.float32(median_width(normalized) * WIDTHS[config["width_index"]]))
    kernel = gaussian_kernel(normalized, normalized, width)
    alpha = ALPHAS[config["alpha_index"]]
    if family == "exact":
        ids = np.arange(len(x))
        beta = exact_coefficients(kernel, yn, weights, (alpha,), exact_backend)[0].astype(np.float32)
    else:
        mode = "rpchol" if family == "project_rpchol" else family.removeprefix("nys_")
        ids, _, _ = select_landmarks(kernel, weights, mode, rank, 17)
        if family == "project_rpchol":
            teacher = exact_coefficients(kernel, yn, weights, (alpha,), exact_backend)[0].astype(np.float32)
            beta = projection_coefficients(kernel, ids, teacher.astype(np.float64))[0].astype(np.float32)
        else:
            beta = nystrom_coefficients(kernel, ids, yn, weights, (alpha,))[0][0].astype(np.float32)
    return {**prep, "centers": normalized[ids].astype(np.float32), "coefficient": beta,
            "width": np.asarray(width, np.float32)}


def measure(run, cache, legacy, output):
    check_lock(run)
    if sha(cache) != CACHE_HASH:
        raise ValueError("only frozen original TRAIN is permitted")
    audit = js(output / "audit.json")
    if not audit["passed"] or audit["selection_sha256"] != sha(run / "selections.json"):
        raise ValueError("a passing audit of the exact choices is required")
    with np.load(cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    role_masks = roles(data["patient"], data["device"])
    records, fit_records = [], []
    for role, (_, held) in role_masks.items():
        query = data["color"][held]
        candidates = [r for r in js(run / "results.json")["records"] if r["role"] == role and r["family"] != "exact_width1_reference"]
        for record in candidates:
            name = filename(record)
            model_path = run / "selected" / role / name
            if sha(model_path) != record["model_sha256"]:
                raise ValueError("selected payload hash mismatch")
            evaluated = nz(run / "evaluated" / role / name)
            np.testing.assert_array_equal(evaluated["row_indices"], np.flatnonzero(held))
            measured = time_queries(nz(model_path), record["family"], query, evaluated["prediction"])
            records.append({"role": role, "family": record["family"], "rank": record["rank"], "seed": record["seed"],
                            "model_sha256": record["model_sha256"], "numeric_bytes": record["numeric_bytes"], **measured})
        # Retimed historical reference implementation, preserving its FP32 math.
        for method, seeds in (("krr", (17,)), ("guided_rbf", (17, 29, 43))):
            for seed in seeds:
                path = legacy / role / "final" / f"{method}_s{seed}.npz"
                receipt = js(path.with_suffix(".json"))
                if sha(path) != receipt["artifact_sha256"]:
                    raise ValueError("legacy reference hash mismatch")
                payload = nz(path)
                measured = time_queries(payload, f"legacy_{method}", query, legacy_predict(payload, query))
                size = sum(v.nbytes for v in payload.values() if np.issubdtype(v.dtype, np.number))
                records.append({"role": role, "family": f"legacy_{method}", "rank": len(payload["centers"]),
                                "seed": seed, "model_sha256": sha(path), "numeric_bytes": size, **measured})
        print(f"TIMED batch-one {role}", flush=True)
    # Exactly one selected seed per fixed family/rank, with complete preparation
    # and its own kernel/landmark/solve cost; no algebra is reused between fits.
    fit, _ = role_masks["mixed"]
    x, y, person, site = (data[k][fit] for k in ("color", "target", "patient", "site"))
    choices = js(run / "selections.json")["roles"]["mixed"]["primary"]
    for family, ranks in choices.items():
        for rank_string, config in ranks.items():
            rank = int(rank_string)
            backends = ("cpu", "cuda") if family in ("exact", "project_rpchol") else ("cpu",)
            expected = nz(run / "selected" / "mixed" / f"{family}_k{rank:03d}_s17.npz")
            for backend in backends:
                samples, max_drift = [], 0.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for repetition in range(4):
                    before = time.perf_counter()
                    actual = standalone_fit(x, y, person, site, family, rank, config, backend)
                    # CUDA helper returns CPU coefficients, so operations and
                    # transfers are already complete before stopping the clock.
                    elapsed = time.perf_counter() - before
                    if repetition:
                        samples.append(elapsed)
                    np.testing.assert_array_equal(actual["centers"], expected["centers"])
                    delta = float(np.abs(predict_kernel(actual, x[:7]) - predict_kernel(expected, x[:7])).max())
                    if delta > .002:
                        raise ValueError(f"standalone refit differs: {family}/{rank}/{backend}: {delta}")
                    max_drift = max(max_drift, delta)
                fit_records.append({"role": "mixed", "n_fit_rows": len(x), "family": family, "rank": rank,
                                    "seed": 17, "exact_backend": backend if family in ("exact", "project_rpchol") else None,
                                    "compact_backend": "cpu", "width_index": config["width_index"], "alpha_index": config["alpha_index"],
                                    "seconds": samples, "median_ms": float(np.median(samples) * 1000),
                                    "maximum_refit_component_drift": max_drift,
                                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()})
        print(f"TIMED standalone {family}", flush=True)
    check_lock(run)
    result = {"source_lock_sha256": sha(run / "source_lock.json"), "selection_sha256": sha(run / "selections.json"),
              "runtime_source_sha256": sha(Path(__file__)), "audit_sha256": sha(output / "audit.json"),
              "hardware": {"cpu": "AMD Ryzen 9 7900X", "processor_identifier": platform.processor(),
                           "gpu": torch.cuda.get_device_name(0), "logical_cpus": os.cpu_count(), "os": platform.platform()},
              "runtime": {"numpy": np.__version__, "torch": torch.__version__, "threads": 1,
                          "kernel_storage": "FP32", "kernel_arithmetic": "FP64; legacy references retain canonical FP32",
                          "query_warmups_per_model": 20, "query_repetitions": 3, "fit_warmups": 1, "fit_repetitions": 3},
              "scope": "Warm batch-one CPU model calls include normalization, kernel evaluation, projections/blends and actual adaptive control. Excludes file load, image/face feature extraction and app/network time. Adaptive initialization is reported separately. Payload bytes are not peak runtime RAM. Measurements are implementation-specific.",
              "fit_scope": "Standalone fixed models on 734 prepared rows, seed17, frozen selected hyperparameters. Includes balancing, fit normalization, direct pairwise median, Gram allocation, landmarks and solve. Projection includes a freshly fitted full teacher. Excludes file loading, model selection, diagnostics for adaptive bounds and persistence. No algebra reused between repetitions. CUDA initialized before warmup.",
              "records": records, "standalone_fit_records": fit_records}
    write_json(output / "runtime.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    measure(args.run, args.cache, args.legacy_run, args.output)


if __name__ == "__main__":
    main()
