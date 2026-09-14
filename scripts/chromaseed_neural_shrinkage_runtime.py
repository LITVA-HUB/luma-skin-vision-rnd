"""Actual NS inference and270 full model reconstructions without cached bases."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from chromaseed_gaussian_runtime import query_time
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_readout_runtime import timing_fit
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    assert not (OUT / "verification.json").exists(), "sealed runtime is read-only"
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    assert sha(args.cache) == CACHE_HASH
    lock, result, audit = (
        js(RUN / "source_lock.json"),
        js(RUN / "results.json"),
        js(OUT / "audit.json"),
    )
    assert audit["passed"] and audit["checks"]["saved_QR_heads"] == 5040
    assert audit["results_sha256"] == sha(RUN / "results.json")
    assert audit["audit_source_sha256"] == sha(
        ROOT / "scripts/chromaseed_neural_shrinkage_audit.py"
    )
    for p, h in {
        **lock["sources"],
        **lock["input_sha256"],
        **audit["dependencies"],
        **audit["artifact_sha256"],
    }.items():
        assert sha(ROOT / p) == h, p
    data = nz(args.cache)
    records, fits = [], []
    for rec in result["records"]:
        mp = RUN / "selected" / rec["role"] / f"{rec['name']}.npz"
        pp = RUN / "evaluated" / rec["role"] / f"{rec['name']}.npz"
        assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        saved = nz(pp)
        records.append(
            dict(
                role=rec["role"],
                group=rec["group"],
                family=rec["family"],
                basis=rec["basis"],
                seed=rec["seed"],
                name=rec["name"],
                numeric_bytes=rec["numeric_bytes"],
                archive_bytes=rec["archive_bytes"],
                numpy_only=query_time(
                    nz(mp), data["color"][saved["row_indices"]], saved["prediction"][0]
                ),
            )
        )
    print("PROFILE381 actual one-row consumers checked", flush=True)
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        cases = [
            r
            for r in result["records"]
            if r["role"] == role
            and r["seed"] == 17
            and r["family"] in ("norm", "perceptual", "fg_norm_static")
        ]
        assert len(cases) == 30
        for i, rec in enumerate(cases, 1):
            model = nz(RUN / "selected" / role / f"{rec['name']}.npz")
            prediction = nz(RUN / "evaluated" / role / f"{rec['name']}.npz")["prediction"][0]
            fits.append(timing_fit(data, rows, rec, model, data["color"][held], prediction))
            if i % 10 == 0:
                print(f"PROFILE {role}:{i}/30 settings x3 full constructions", flush=True)
    assert (len(records), len(fits)) == (381, 90)
    dependencies = (
        "scripts/chromaseed_neural_readout_runtime.py",
        "scripts/chromaseed_gaussian_runtime.py",
        "scripts/chromaseed_neural_readout.py",
        "scripts/chromaseed_gaussian.py",
        "scripts/chromaseed_feature_groups.py",
        "scripts/chromaseed_gaussian_numpy.py",
        "scripts/chromaseed_feature_groups_numpy.py",
    )
    write_json(
        OUT / "runtime.json",
        dict(
            source_lock_sha256=sha(RUN / "source_lock.json"),
            selection_sha256=sha(RUN / "selections.json"),
            results_sha256=sha(RUN / "results.json"),
            audit_sha256=sha(OUT / "audit.json"),
            runtime_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in dependencies},
            records=records,
            standalone_fit_records=fits,
            complete_fits_including_warmups=270,
            exact_payload_repeats=sum(r["exact_payload_repeats"] for r in fits),
            wall_seconds=time.perf_counter() - started,
            scope="CPU1thread.381 actual consumers;20 warmups/3 query passes.84 seed17 heads and6 FG settings x3 complete fits, first warmup discarded. Every hidden representation reconstructed from scratch. Fit includes all representation epochs, feature moments, weighted metric/readout and export. Prepared color features only; no image/I/O/import/phone/GPU inference claim. Explicit array/state bytes are not process peak RAM.",
        ),
    )
    print(
        dict(
            passed=True,
            full_fits=270,
            exact=sum(r["exact_payload_repeats"] for r in fits),
            wall_seconds=time.perf_counter() - started,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
