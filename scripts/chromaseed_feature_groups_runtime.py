"""Time real FG batch-one consumers and 384 complete matching fits, sequentially."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact
from chromaseed_feature_groups import fit_single
from chromaseed_feature_groups_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from chromaseed_projection import fit_single as x_fit_single
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def query_time(model, x, expected):
    start = time.perf_counter_ns()
    call = Predictor(model)
    initialization = (time.perf_counter_ns() - start) / 1000
    observed = np.array([call(v) for v in x])
    drift = float(np.abs(observed - expected).max())
    assert drift <= 2e-8
    for i in range(20):
        call(x[i % len(x)])
    times, enabled = [], gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            for v in x:
                start = time.perf_counter_ns()
                call(v)
                times.append((time.perf_counter_ns() - start) / 1000)
    finally:
        if enabled:
            gc.enable()
    return dict(
        median_us=float(np.median(times)),
        p95_us=float(np.quantile(times, 0.95)),
        query_count=len(times),
        verification_rows=len(x),
        max_lab_drift=drift,
        initialization_us=initialization,
        cached_array_bytes=call.cached_array_bytes,
    )


def timing_fit(data, rows, rec, expected):
    times = []
    args = [data[k][rows] for k in ("color", "target", "patient", "site", "device")]
    for repeat in range(4):
        start = time.perf_counter()
        if rec["group"] == "projected16":
            model, _ = x_fit_single(*args, rec["family"], 17, "d16_t05", rec["alpha"])
        else:
            model, _ = fit_single(*args, rec["family"], 17, rec["group"], rec["alpha"])
        elapsed = time.perf_counter() - start
        exact(model, expected)
        if repeat:
            times.append(elapsed)
    return dict(
        role=rec["role"],
        family=rec["family"],
        group=rec["group"],
        seed=17,
        name=rec["name"],
        alpha=rec["alpha"],
        n_fit_rows=len(rows),
        median_seconds=float(np.median(times)),
        seconds=times,
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, out, start = args.run, args.output, time.perf_counter()
    lock, result = js(run / "source_lock.json"), js(run / "results.json")
    audit = js(out / "audit.json")
    assert audit["passed"] and audit["checks"]["qr_refits"] == 288
    assert (
        audit["source_lock_sha256"] == result["source_lock_sha256"] == sha(run / "source_lock.json")
    )
    assert audit["selection_sha256"] == sha(run / "selections.json")
    assert audit["results_sha256"] == sha(run / "results.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_feature_groups_audit.py")
    for p, h in {**lock["sources"], **lock["input_sha256"], **audit["dependencies"]}.items():
        assert sha(ROOT / p) == h, p
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    records, fits = [], []
    for rec in result["records"]:
        mp = run / "selected" / rec["role"] / f"{rec['name']}.npz"
        pp = run / "evaluated" / rec["role"] / f"{rec['name']}.npz"
        assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        saved = nz(pp)
        records.append(
            dict(
                role=rec["role"],
                family=rec["family"],
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
    print("PROFILE291 actual FG NumPy consumers checked", flush=True)
    for role, (fit, _) in masks.items():
        rows = np.flatnonzero(fit)
        for rec in (r for r in result["records"] if r["role"] == role and r["seed"] == 17):
            model = nz(run / "selected" / role / f"{rec['name']}.npz")
            fits.append(timing_fit(data, rows, rec, model))
        print(f"PROFILE {role}:32 settings x4 complete fits checked", flush=True)
    assert len(records) == 291 and len(fits) == 96
    deps = (
        "scripts/chromaseed_feature_groups.py",
        "scripts/chromaseed_feature_groups_numpy.py",
        "scripts/chromaseed_projection.py",
        "scripts/chromaseed_projection_numpy.py",
        "scripts/chromaseed_affine_audit.py",
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
            complete_fits_including_warmups=384,
            wall_seconds=time.perf_counter() - start,
            scope="291 actual one-row consumers,20 warmups and3 passes;96 settings x4 complete fits, including warmups. One CPU thread. Fit includes normalizers, weights, exact width, landmarks, gate when active, perceptual metric and readout. X control includes its projection. Excludes process imports/I/O/grid/image decoding and skin extraction. Numeric/cache/archive storage reported separately; cache excludes temporaries, Python/process overhead and caller payload.",
        ),
    )
    print("FG RUNTIME PASS:384 complete fits with exact payloads", flush=True)


if __name__ == "__main__":
    main()
