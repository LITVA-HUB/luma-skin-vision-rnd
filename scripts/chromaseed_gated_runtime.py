"""Sequential canonical/standalone inference and complete-fit timing after G audit."""

from __future__ import annotations

import argparse
import gc
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_gated import fit_single, model_id, predict, unpack
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def query_time(model, query, expected, standalone):
    start = time.perf_counter_ns()
    call = Predictor(model) if standalone else lambda x: predict(model, x[None])[0]
    init = (time.perf_counter_ns() - start) / 1000.0
    output = np.array([call(x) for x in query])
    drift = float(np.abs(output - expected).max())
    assert drift < 2e-8
    for i in range(20):
        call(query[i % len(query)])
    times = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            for x in query:
                before = time.perf_counter_ns()
                call(x)
                times.append((time.perf_counter_ns() - before) / 1000.0)
    finally:
        if enabled:
            gc.enable()
    return dict(
        median_us=float(np.median(times)),
        p95_us=float(np.quantile(times, 0.95)),
        query_count=len(times),
        verification_rows=len(query),
        max_lab_drift=drift,
        initialization_us=init,
        cached_array_bytes=call.cached_array_bytes if standalone else None,
    )


def timing_fit(data, rows, family, alpha, rho, expected):
    times = []
    detail = []
    for i in range(4):
        start = time.perf_counter()
        model, info = fit_single(
            *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
            family,
            17,
            alpha,
            rho,
        )
        elapsed = time.perf_counter() - start
        assert set(model) == set(expected)
        for key in model:
            np.testing.assert_array_equal(model[key], expected[key])
        if i:
            times.append(elapsed)
            detail.append(info)
    return dict(
        family=family,
        residual_lambda=alpha,
        rho=rho,
        seed=17,
        n_fit_rows=len(rows),
        median_seconds=float(np.median(times)),
        seconds=times,
        details=detail,
    )


def main():
    parser = argparse.ArgumentParser()
    for key in ("run", "cache", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    out = args.output
    audit = js(out / "audit.json")
    lock = js(run / "source_lock.json")
    assert (
        audit["passed"]
        and audit["source_lock_sha256"] == sha(run / "source_lock.json")
        and audit["selection_sha256"] == sha(run / "selections.json")
    )
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_gated_audit.py")
    for path, h in {**lock["sources"], **audit["dependencies"]}.items():
        assert sha(ROOT / path) == h, path
    assert sha(args.cache) == CACHE_HASH
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    results = js(run / "results.json")
    records = []
    fits = []
    fixed = []
    for record in results["records"]:
        role, family, seed = record["role"], record["family"], record["seed"]
        name = f"{family}_s{seed}.npz"
        mp, pp = run / "selected" / role / name, run / "evaluated" / role / name
        assert sha(mp) == record["model_sha256"] and sha(pp) == record["prediction_sha256"]
        model, saved = nz(mp), nz(pp)
        query = data["color"][saved["row_indices"]]
        records.append(
            dict(
                role=role,
                family=family,
                seed=seed,
                numeric_bytes=record["numeric_bytes"],
                active_gate=record["active_gate"],
                canonical=query_time(model, query, saved["prediction"], False),
                numpy_only=query_time(model, query, saved["prediction"], True),
            )
        )
    for role, (fit, held) in masks.items():
        rows = np.flatnonzero(fit)
        for record in (r for r in results["records"] if r["role"] == role and r["seed"] == 17):
            expected = nz(run / "selected" / role / f"{record['family']}_s17.npz")
            fits.append(
                dict(
                    role=role,
                    **timing_fit(
                        data,
                        rows,
                        record["family"],
                        record["residual_lambda"],
                        record["rho"],
                        expected,
                    ),
                )
            )
        print(
            f"PROFILE {role}: eight complete selected fits, warmup plus3 repeats each", flush=True
        )
    fit, held = masks["mixed"]
    rows = np.flatnonzero(fit)
    query = data["color"][held]
    bank = nz(run / "final/mixed/bank/models.npz")
    for family in ("norm_soft", "norm_hard", "perceptual_soft", "perceptual_hard"):
        model = unpack(bank, model_id(family, 17, 0.1, 1.0))
        assert "correction" in model
        expected = predict(model, query)
        fixed.append(
            dict(
                role="mixed",
                family=family,
                residual_lambda=0.1,
                rho=1.0,
                numeric_bytes=sum(v.nbytes for v in model.values()),
                canonical=query_time(model, query, expected, False),
                numpy_only=query_time(model, query, expected, True),
                standalone_fit=timing_fit(data, rows, family, 0.1, 1.0, model),
            )
        )
    deps = (
        "scripts/chromaseed_gated.py",
        "scripts/chromaseed_gated_numpy.py",
        "scripts/chromaseed_kernel_audit.py",
        "tests/test_chromaseed_gated_numpy.py",
    )
    write_json(
        out / "runtime.json",
        dict(
            source_lock_sha256=sha(run / "source_lock.json"),
            selection_sha256=sha(run / "selections.json"),
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
            fixed_active_records=fixed,
            selected_complete_fits_including_warmups=96,
            fixed_complete_fits_including_warmups=16,
            scope="Actual batch-one canonical and NumPy-only predictors;20 warmups/3 query passes. All72 outputs verified. Complete fits include weights/preparation/gate/basis/readout; no I/O/import/search. Numeric payload excludes runtime cache/library. Four active cost probes do not generate new quality scores.",
        ),
    )


if __name__ == "__main__":
    main()
