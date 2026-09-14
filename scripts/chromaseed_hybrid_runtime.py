"""Sequential actual H NumPy consumers and complete single-model timing fits."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact
from chromaseed_gated_audit import model_from
from chromaseed_hybrid import fit_single, model_id
from chromaseed_hybrid_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def query_time(model, x, expected):
    start = time.perf_counter_ns()
    call = Predictor(model)
    initialization = (time.perf_counter_ns() - start) / 1000
    output = np.array([call(v) for v in x])
    drift = float(np.abs(output - expected).max())
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


def timing_fit(data, rows, loss, setting, expected):
    times = []
    for repeat in range(4):
        start = time.perf_counter()
        m, _ = fit_single(
            *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
            loss,
            17,
            **setting,
        )
        elapsed = time.perf_counter() - start
        exact(m, expected)
        if repeat:
            times.append(elapsed)
    return dict(
        loss=loss,
        **setting,
        seed=17,
        n_fit_rows=len(rows),
        median_seconds=float(np.median(times)),
        seconds=times,
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    audit, lock, result = (
        js(out / "audit.json"),
        js(run / "source_lock.json"),
        js(run / "results.json"),
    )
    assert audit["passed"] and audit["source_lock_sha256"] == sha(run / "source_lock.json")
    assert audit["selection_sha256"] == sha(run / "selections.json") and audit[
        "results_sha256"
    ] == sha(run / "results.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_hybrid_audit.py")
    for p, h in {**lock["sources"], **lock["input_sha256"], **audit["dependencies"]}.items():
        assert sha(ROOT / p) == h, p
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert all(
        os.environ[k] == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    records, fits, fixed = [], [], []
    for rec in result["records"]:
        mp, pp = (
            run / "selected" / rec["role"] / f"{rec['name']}.npz",
            run / "evaluated" / rec["role"] / f"{rec['name']}.npz",
        )
        assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        saved = nz(pp)
        records.append(
            dict(
                role=rec["role"],
                family=rec["family"],
                seed=rec["seed"],
                name=rec["name"],
                numeric_bytes=rec["numeric_bytes"],
                active_hybrid=rec["active_hybrid"],
                numpy_only=query_time(
                    nz(mp), data["color"][saved["row_indices"]], saved["prediction"][0]
                ),
            )
        )
    print("PROFILE111 actual NumPy consumers passed", flush=True)
    for role, (fit, _) in masks.items():
        rows = np.flatnonzero(fit)
        for rec in (
            r
            for r in result["records"]
            if r["role"] == role and r["policy"] == "inner" and r["seed"] == 17
        ):
            setting = {k: rec[k] for k in ("kind", "alpha", "rho", "power")}
            m = nz(run / "selected" / role / f"{rec['name']}.npz")
            fits.append(
                dict(
                    role=role,
                    family=rec["family"],
                    **timing_fit(data, rows, rec["loss"], setting, m),
                )
            )
        print(f"PROFILE {role}:10 selected settings x4 full fits", flush=True)
    rows, bank = np.flatnonzero(masks["mixed"][0]), nz(run / "final/mixed/bank/models.npz")
    for loss in ("norm", "perceptual"):
        for kind in ("blend", "uniform", "support"):
            setting = dict(kind=kind, alpha=0.1, rho=0.5, power=4 if kind == "support" else 0)
            m = model_from(bank, model_id(loss, 17, **setting))
            fixed.append(dict(role="mixed", **timing_fit(data, rows, loss, setting, m)))
    assert len(records) == 111 and len(fits) == 30 and len(fixed) == 6
    deps = (
        "scripts/chromaseed_hybrid.py",
        "scripts/chromaseed_hybrid_numpy.py",
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_gated_audit.py",
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
            fixed_positive_records=fixed,
            selected_complete_fits_including_warmups=120,
            fixed_complete_fits_including_warmups=24,
            scope="Actual111 one-record NumPy consumers,20 warmups/3 passes,144 complete fits including warmups. Includes original weights/moments/exact widths/raw landmarks/gate/readout, plus projection/shared projected basis/branch when needed. Single projected fit also computes raw backbone readout. Excludes imports, I/O, grid search and image/skin extraction. Cached arrays exclude process, temporaries and retained caller objects. One CPU thread, no GPU claim.",
        ),
    )
    print("H RUNTIME PASS:144 complete fits, exact payloads", flush=True)


if __name__ == "__main__":
    main()
