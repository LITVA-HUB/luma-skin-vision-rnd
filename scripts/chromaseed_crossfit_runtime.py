"""Actual unchanged H consumer, including full teacher-pool costs in C fits."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact
from chromaseed_crossfit import fit_single as c_fit
from chromaseed_hybrid import fit_single as h_fit
from chromaseed_hybrid_runtime import query_time
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def timing_fit(data, rows, rec, expected):
    setting = {k: rec[k] for k in ("kind", "alpha", "rho", "power")}
    times, details = [], []
    for repeat in range(4):
        args = [data[k][rows] for k in ("color", "target", "patient", "site", "device")]
        start = time.perf_counter()
        if rec["origin"] == "C":
            m, info = c_fit(*args, rec["loss"], 17, rec["arm"], setting)
        else:
            m, info = h_fit(*args, rec["loss"], 17, **setting)
        elapsed = time.perf_counter() - start
        exact(m, expected)
        if repeat:
            times.append(elapsed)
            details.append(info)
    return dict(
        role=rec["role"],
        family=rec["family"],
        origin=rec["origin"],
        loss=rec["loss"],
        seed=17,
        n_fit_rows=len(rows),
        **setting,
        median_seconds=float(np.median(times)),
        seconds=times,
        details=details,
    )


def main():
    parser = argparse.ArgumentParser()
    for k in ("run", "cache", "output"):
        parser.add_argument("--" + k, type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    audit, lock, result = (
        js(out / "audit.json"),
        js(run / "source_lock.json"),
        js(run / "results.json"),
    )
    assert audit["passed"] and audit["source_lock_sha256"] == sha(run / "source_lock.json")
    assert audit["settings_sha256"] == sha(run / "frozen_settings.json") and audit[
        "results_sha256"
    ] == sha(run / "results.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_crossfit_audit.py")
    for p, h in {**lock["sources"], **lock["input_sha256"], **audit["dependencies"]}.items():
        assert sha(ROOT / p) == h, p
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert all(
        os.environ[k] == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    queries, fits = [], []
    for rec in result["records"]:
        mp, pp = (
            run / "selected" / rec["role"] / f"{rec['name']}.npz",
            run / "evaluated" / rec["role"] / f"{rec['name']}.npz",
        )
        assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        saved = nz(pp)
        queries.append(
            dict(
                role=rec["role"],
                family=rec["family"],
                origin=rec["origin"],
                seed=rec["seed"],
                name=rec["name"],
                numeric_bytes=rec["numeric_bytes"],
                numpy_only=query_time(
                    nz(mp), data["color"][saved["row_indices"]], saved["prediction"][0]
                ),
            )
        )
    print("PROFILE C:183 actual NumPy consumers passed", flush=True)
    for role, (fit, _) in masks.items():
        rows = np.flatnonzero(fit)
        cases = [
            r
            for r in result["records"]
            if r["role"] == role and r["seed"] == 17 and r["kind"] not in ("reference", "constant")
        ]
        assert len(cases) == 18
        for rec in cases:
            m = nz(run / "selected" / role / f"{rec['name']}.npz")
            fits.append(timing_fit(data, rows, rec, m))
        print(f"PROFILE C {role}:72 complete fits including required teachers passed", flush=True)
    assert len(queries) == 183 and len(fits) == 54
    assert sum(r["origin"] == "C" for r in fits) == 24
    deps = (
        "scripts/chromaseed_crossfit.py",
        "scripts/chromaseed_hybrid.py",
        "scripts/chromaseed_hybrid_runtime.py",
        "scripts/chromaseed_hybrid_numpy.py",
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_kernel_audit.py",
    )
    write_json(
        out / "runtime.json",
        dict(
            source_lock_sha256=sha(run / "source_lock.json"),
            settings_sha256=sha(run / "frozen_settings.json"),
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
            records=queries,
            full_fit_records=fits,
            C_complete_fits_including_warmups=96,
            H_complete_fits_including_warmups=120,
            scope="183 actual one-record consumers;20 warmups/3 passes.216 complete fits,96 C include n-person excluded-teacher pool plus native tables/routing and full student geometry/readout/export;120 same-run H controls. No imports/I/O/grid selection/image extraction/phone or GPU timing. Cached arrays exclude process, temporaries and retained caller objects. One CPU thread, no concurrent heavy work.",
        ),
    )
    print("C runtime PASS:216 complete exact-payload fits", flush=True)


if __name__ == "__main__":
    main()
