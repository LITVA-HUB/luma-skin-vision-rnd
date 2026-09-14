"""A sequential standalone queries and112 complete fit repetitions after audit."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_affine import FAMILIES, fit_single, model_id
from chromaseed_gated_audit import model_from
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


class Constant:
    """Only the fixed native-Lab negative control, with the same input checks."""

    def __init__(self, model):
        assert set(model) == {"constant_lab"}
        value = model["constant_lab"]
        assert value.shape == (3,) and value.dtype == np.float32 and np.isfinite(value).all()
        self.value = value.astype(np.float64)

    @property
    def cached_array_bytes(self):
        return self.value.nbytes

    def __call__(self, color):
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        return self.value.copy()


def query_time(model, query, expected):
    before = time.perf_counter_ns()
    call = Constant(model) if "constant_lab" in model else Predictor(model)
    init = (time.perf_counter_ns() - before) / 1000
    output = np.array([call(v) for v in query])
    drift = float(np.abs(output - expected).max())
    assert drift <= 2e-8
    for i in range(20):
        call(query[i % len(query)])
    times = []
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(3):
            for v in query:
                before = time.perf_counter_ns()
                call(v)
                times.append((time.perf_counter_ns() - before) / 1000)
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
        cached_array_bytes=call.cached_array_bytes,
    )


def timing_fit(data, rows, family, alpha, eta, expected):
    times, details = [], []
    for repeat in range(4):
        before = time.perf_counter()
        model, info = fit_single(
            *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
            family,
            17,
            alpha,
            eta,
        )
        elapsed = time.perf_counter() - before
        assert set(model) == set(expected)
        for key in model:
            np.testing.assert_array_equal(model[key], expected[key])
        if repeat:
            times.append(elapsed)
            details.append(info)
    return dict(
        family=family,
        alpha=alpha,
        eta=eta,
        seed=17,
        n_fit_rows=len(rows),
        median_seconds=float(np.median(times)),
        seconds=times,
        details=details,
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
    assert audit["selection_sha256"] == sha(run / "selections.json")
    assert audit["results_sha256"] == sha(run / "results.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_affine_audit.py")
    for path, h in {**lock["sources"], **lock["input_sha256"], **audit["dependencies"]}.items():
        assert sha(ROOT / path) == h, path
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[key] == "1"
    with np.load(args.cache, allow_pickle=False) as arr:
        data = {k: arr[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    records, fits, fixed = [], [], []
    for rec in result["records"]:
        mp = run / "selected" / rec["role"] / f"{rec['name']}.npz"
        pp = run / "evaluated" / rec["role"] / f"{rec['name']}.npz"
        assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        saved = nz(pp)
        records.append(
            dict(
                role=rec["role"],
                family=rec["family"],
                policy=rec["policy"],
                seed=rec["seed"],
                name=rec["name"],
                numeric_bytes=rec["numeric_bytes"],
                active_gate=rec["active_gate"],
                numpy_only=query_time(
                    nz(mp), data["color"][saved["row_indices"]], saved["prediction"][0]
                ),
            )
        )
    print(
        "PROFILE all111 standalone consumers: outputs verified,20 warmups/3 passes each", flush=True
    )
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
                    **timing_fit(data, rows, rec["family"], rec["alpha"], rec["eta"], m),
                )
            )
        print(f"PROFILE {role}:8 selected settings x4 complete parity-checked fits", flush=True)
    bank = nz(run / "final/mixed/bank/models.npz")
    rows = np.flatnonzero(masks["mixed"][0])
    for family in FAMILIES:
        m = model_from(bank, model_id(family, 17, 0.1, 0.75))
        fixed.append(dict(role="mixed", **timing_fit(data, rows, family, 0.1, 0.75, m)))
    assert len(records) == 111 and len(fits) == 24 and len(fixed) == 4
    deps = (
        "scripts/chromaseed_affine.py",
        "scripts/chromaseed_gated_audit.py",
        "scripts/chromaseed_gated_numpy.py",
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
            scope="111 actual standalone batch-one consumers,20 warmups/3 passes.112 exact full fits including warmups, weights/normalization/bandwidth/landmarks/gate/augmentation/readout. Excludes imports, I/O, search and image feature preparation. CPU one thread, numeric payload and cached arrays exclude library/process memory. Constant control uses a checked NumPy-only broadcast consumer.",
        ),
    )
    print("A RUNTIME PASS:112 exact full fits", flush=True)


if __name__ == "__main__":
    main()
