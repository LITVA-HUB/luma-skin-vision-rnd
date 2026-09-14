"""Actual standalone X response and complete-fit cost, sequential after audit."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_projection import FAMILIES, fit_single, model_id
from chromaseed_projection_numpy import Predictor
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def query_time(model, x, expected):
    start = time.perf_counter_ns()
    call = Predictor(model)
    init = (time.perf_counter_ns() - start) / 1000
    out = np.array([call(v) for v in x])
    drift = float(np.abs(out - expected).max())
    assert drift <= 2e-8
    for i in range(20):
        call(x[i % len(x)])
    times = []
    enabled = gc.isenabled()
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
        initialization_us=init,
        cached_array_bytes=call.cached_array_bytes,
    )


def timing_fit(data, rows, family, representation, alpha, expected):
    times, detail = [], []
    for repeat in range(4):
        start = time.perf_counter()
        model, info = fit_single(
            *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
            family,
            17,
            representation,
            alpha,
        )
        elapsed = time.perf_counter() - start
        assert set(model) == set(expected)
        for key in model:
            np.testing.assert_array_equal(model[key], expected[key])
        if repeat:
            times.append(elapsed)
            detail.append(info)
    return dict(
        family=family,
        representation=representation,
        alpha=alpha,
        seed=17,
        n_fit_rows=len(rows),
        median_seconds=float(np.median(times)),
        seconds=times,
        details=detail,
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
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_projection_audit.py")
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
                policy=rec["policy"],
                seed=rec["seed"],
                name=rec["name"],
                representation=rec.get("representation", "reference"),
                numeric_bytes=rec["numeric_bytes"],
                active_gate=rec["active_gate"],
                latent_dimension=rec["latent_dimension"],
                numpy_only=query_time(
                    nz(mp), data["color"][saved["row_indices"]], saved["prediction"][0]
                ),
            )
        )
    print("PROFILE111 actual NumPy consumers:20 warmups/3 passes, all outputs pass", flush=True)
    for role, (fit, _) in masks.items():
        rows = np.flatnonzero(fit)
        for rec in (
            r
            for r in result["records"]
            if r["role"] == role and r["family"] in FAMILIES and r["seed"] == 17
        ):
            m = nz(run / "selected" / role / f"{rec['name']}.npz")
            fits.append(
                dict(
                    role=role,
                    policy=rec["policy"],
                    **timing_fit(data, rows, rec["family"], rec["representation"], rec["alpha"], m),
                )
            )
        print(f"PROFILE {role}:8 choices x4 complete exact fits", flush=True)
    rows = np.flatnonzero(masks["mixed"][0])
    bank = nz(run / "final/mixed/bank/models.npz")
    for family in FAMILIES:
        m = model_from(bank, model_id(family, 17, "d8_t01", 0.1))
        fixed.append(dict(role="mixed", **timing_fit(data, rows, family, "d8_t01", 0.1, m)))
    assert len(records) == 111 and len(fits) == 24 and len(fixed) == 4
    deps = (
        "scripts/chromaseed_projection.py",
        "scripts/chromaseed_projection_numpy.py",
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
            selected_complete_fits_including_warmups=96,
            fixed_complete_fits_including_warmups=16,
            scope="Actual111 batch-one NumPy consumers and112 exact full fits including warmups. Full fit includes original weights/moments/covariance/projection/exact width/landmarks/gate/readout. No imports/I/O/search/image preparation. Cached arrays exclude process, transient and retained caller objects. CPU one thread.",
        ),
    )
    print("X RUNTIME PASS:112 exact full fits", flush=True)


if __name__ == "__main__":
    main()
