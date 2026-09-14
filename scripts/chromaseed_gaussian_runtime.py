"""Time75 real consumers and72 complete individual TG/FG fits after audit."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_feature_groups import fit_single as fg_fit
from chromaseed_feature_groups_numpy import Predictor as FGPredictor
from chromaseed_gaussian import fit_single
from chromaseed_gaussian_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def consumer(model):
    return Predictor(model) if "w1" in model else FGPredictor(model)


def query_time(model, x, expected):
    started = time.perf_counter_ns()
    call = consumer(model)
    initialization = (time.perf_counter_ns() - started) / 1000
    output = np.array([call(row) for row in x])
    drift = float(np.max(np.abs(output - expected)))
    assert drift <= 2e-8
    for i in range(20):
        call(x[i % len(x)])
    durations, enabled = [], gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            for row in x:
                started = time.perf_counter_ns()
                call(row)
                durations.append((time.perf_counter_ns() - started) / 1000)
    finally:
        if enabled:
            gc.enable()
    return dict(
        median_us=float(np.median(durations)),
        p95_us=float(np.quantile(durations, 0.95)),
        query_count=len(durations),
        verification_rows=len(x),
        max_lab_drift=drift,
        initialization_us=initialization,
        cached_array_bytes=call.cached_array_bytes,
    )


def timing_fit(data, rows, rec, expected, x_query, y_expected):
    args = [data[k][rows] for k in ("color", "target", "patient", "site", "device")]
    times, max_parameter, max_output = [], 0.0, 0.0
    exact_runs = 0
    training_bytes = None
    for repeat in range(3):
        started = time.perf_counter()
        if rec["method"] == "fg_norm_static":
            model, _ = fg_fit(*args, "norm_static", 17, rec["group"], 0.1)
        else:
            model, metadata = fit_single(
                *args[:4], rec["group"], rec["method"], rec["parameter"], 17, rec["epoch"]
            )
            training_bytes = metadata["training_parameter_state_bytes"]
        elapsed = time.perf_counter() - started
        assert set(model) == set(expected)
        all_exact = True
        for key in model:
            assert model[key].dtype == expected[key].dtype
            difference = float(np.abs(model[key].astype(float) - expected[key].astype(float)).max())
            max_parameter = max(max_parameter, difference)
            np.testing.assert_allclose(model[key], expected[key], atol=2e-6, rtol=2e-6)
            all_exact &= np.array_equal(model[key], expected[key])
        exact_runs += int(all_exact)
        call = consumer(model)
        output = np.array([call(row) for row in x_query])
        difference = float(np.abs(output - y_expected).max())
        max_output = max(max_output, difference)
        assert difference <= 0.001
        if repeat:
            times.append(elapsed)
    return dict(
        role=rec["role"],
        method=rec["method"],
        group=rec["group"],
        seed=17,
        name=rec["name"],
        parameter=rec["parameter"],
        epoch=rec["epoch"],
        n_fit_rows=len(rows),
        median_seconds=float(np.median(times)),
        seconds=times,
        complete_fits_including_warmup=3,
        exact_payload_repeats=exact_runs,
        max_parameter_absolute=max_parameter,
        max_lab_drift=max_output,
        training_parameter_state_bytes=training_bytes,
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, out, started = args.run, args.output, time.perf_counter()
    lock, result = js(run / "source_lock.json"), js(run / "results.json")
    audit = js(out / "audit.json")
    assert audit["passed"] and audit["checks"]["scalar_refit_trajectories"] == 54
    assert (
        audit["source_lock_sha256"] == result["source_lock_sha256"] == sha(run / "source_lock.json")
    )
    assert audit["selection_sha256"] == result["selection_sha256"] == sha(run / "selections.json")
    assert audit["results_sha256"] == sha(run / "results.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_gaussian_audit.py")
    for p, h in {**lock["sources"], **lock["input_sha256"], **audit["dependencies"]}.items():
        assert sha(ROOT / p) == h, p
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    records, fits = [], []
    for rec in result["records"]:
        mp, pp = (
            run / "models" / rec["role"] / f"{rec['name']}.npz",
            run / "evaluated" / rec["role"] / f"{rec['name']}.npz",
        )
        assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        saved = nz(pp)
        records.append(
            dict(
                role=rec["role"],
                method=rec["method"],
                group=rec["group"],
                seed=rec["seed"],
                name=rec["name"],
                numeric_bytes=rec["numeric_bytes"],
                archive_bytes=rec["archive_bytes"],
                numpy_only=query_time(
                    nz(mp), data["color"][saved["row_indices"]], saved["prediction"][0]
                ),
            )
        )
    print("PROFILE75 actual TG/FG consumers checked", flush=True)
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        for rec in (r for r in result["records"] if r["role"] == role and r["seed"] == 17):
            model = nz(run / "models" / role / f"{rec['name']}.npz")
            pred = nz(run / "evaluated" / role / f"{rec['name']}.npz")["prediction"][0]
            fits.append(timing_fit(data, rows, rec, model, data["color"][held], pred))
            print(f"PROFILE {role}/{rec['method']}/{rec['group']}:3 full fits checked", flush=True)
    assert len(records) == 75 and len(fits) == 24
    deps = (
        "scripts/chromaseed_gaussian.py",
        "scripts/chromaseed_gaussian_numpy.py",
        "scripts/chromaseed_feature_groups.py",
        "scripts/chromaseed_feature_groups_numpy.py",
        "scripts/chromaseed_kernel_audit.py",
        "scripts/skin_local_search_train.py",
    )
    write_json(
        out / "runtime.json",
        dict(
            source_lock_sha256=sha(run / "source_lock.json"),
            selection_sha256=sha(run / "selections.json"),
            results_sha256=sha(run / "results.json"),
            audit_sha256=sha(out / "audit.json"),
            runtime_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in deps},
            hardware=dict(
                platform=platform.platform(),
                processor=platform.processor(),
                threads=1,
                numpy=np.__version__,
            ),
            records=records,
            standalone_fit_records=fits,
            complete_fits_including_warmups=72,
            new_network_complete_fits=54,
            FG_complete_fits=18,
            wall_seconds=time.perf_counter() - started,
            scope="75 actual one-row consumers,20 warmups and3 passes;24 seed17 settings x3 complete fits, one warmup and two measured. One CPU thread. New full fit includes normalizers, balanced weights, initialization, orders, all selected-epoch updates and export; FG includes fresh kernel width, centers and readout. Primary sweep batching, process imports, file I/O and image preprocessing excluded. Training-state bytes count persistent parameter/optimizer arrays only; caller data, transient arrays, RNG and Python/process memory are separate.",
        ),
    )
    print("TG RUNTIME PASS:72 complete matching fits", flush=True)


if __name__ == "__main__":
    main()
