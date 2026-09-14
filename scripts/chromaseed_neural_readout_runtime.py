"""Actual NR consumer profiling and full, uncached model construction repeats."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
from chromaseed_feature_groups import fit_single as fg_fit
from chromaseed_gaussian_runtime import consumer, query_time
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_readout import fit_single
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def timing_fit(data, rows, rec, expected, x_query, y_expected):
    args = [data[k][rows] for k in ("color", "target", "patient", "site", "device")]
    times, exact_runs, max_parameter, max_output = [], 0, 0.0, 0.0
    training_bytes, head_arrays = None, None
    for repeat in range(3):
        start = time.perf_counter()
        if rec["family"] == "fg_norm_static":
            model, _ = fg_fit(*args, "norm_static", 17, rec["group"], 0.1)
        else:
            model, metadata = fit_single(
                *args[:4], rec["group"], rec["basis"], rec["family"], rec["alpha"], 17
            )
            if metadata["representation"] is not None:
                training_bytes = metadata["representation"]["training_parameter_state_bytes"]
            if metadata["readout"] is not None:
                head_arrays = {
                    k: metadata["readout"][k]
                    for k in (
                        "temporary_design_bytes",
                        "temporary_metric_bytes",
                        "temporary_system_bytes",
                    )
                }
        elapsed = time.perf_counter() - start
        assert set(model) == set(expected)
        identical = True
        for key in model:
            assert model[key].dtype == expected[key].dtype
            diff = float(np.abs(model[key].astype(float) - expected[key].astype(float)).max())
            max_parameter = max(max_parameter, diff)
            np.testing.assert_allclose(model[key], expected[key], atol=2e-6, rtol=2e-6)
            identical &= np.array_equal(model[key], expected[key])
        exact_runs += int(identical)
        call = consumer(model)
        output = np.array([call(row) for row in x_query])
        diff = float(np.abs(output - y_expected).max())
        max_output = max(max_output, diff)
        assert diff <= 0.001
        if repeat:
            times.append(elapsed)
    return dict(
        role=rec["role"],
        family=rec["family"],
        basis=rec["basis"],
        group=rec["group"],
        seed=17,
        name=rec["name"],
        alpha=rec["alpha"],
        n_fit_rows=len(rows),
        median_seconds=float(np.median(times)),
        seconds=times,
        complete_fits_including_warmup=3,
        exact_payload_repeats=exact_runs,
        max_parameter_absolute=max_parameter,
        max_lab_drift=max_output,
        representation_training_parameter_state_bytes=training_bytes,
        head_training_array_bytes=head_arrays,
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, out, started = args.run, args.output, time.perf_counter()
    assert not (out / "verification.json").exists(), "sealed profiling is read-only"
    lock, result, audit = (
        js(run / "source_lock.json"),
        js(run / "results.json"),
        js(out / "audit.json"),
    )
    assert audit["passed"] and audit["checks"]["qr_head_refits"] == 3024
    assert (
        audit["source_lock_sha256"] == result["source_lock_sha256"] == sha(run / "source_lock.json")
    )
    assert audit["selection_sha256"] == result["selection_sha256"] == sha(run / "selections.json")
    assert audit["results_sha256"] == sha(run / "results.json")
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_neural_readout_audit.py")
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
            run / "selected" / rec["role"] / f"{rec['name']}.npz",
            run / "evaluated" / rec["role"] / f"{rec['name']}.npz",
        )
        assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        saved = nz(pp)
        records.append(
            dict(
                role=rec["role"],
                family=rec["family"],
                basis=rec["basis"],
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
    print("PROFILE381 actual TG/FG consumers checked", flush=True)
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        cases = [r for r in result["records"] if r["role"] == role and r["seed"] == 17]
        assert len(cases) == 42
        for i, rec in enumerate(cases, 1):
            model = nz(run / "selected" / role / f"{rec['name']}.npz")
            pred = nz(run / "evaluated" / role / f"{rec['name']}.npz")["prediction"][0]
            fits.append(timing_fit(data, rows, rec, model, data["color"][held], pred))
            if i % 7 == 0:
                print(f"PROFILE {role}:{i}/42 settings x3 complete fits checked", flush=True)
    assert len(records) == 381 and len(fits) == 126
    deps = (
        "scripts/chromaseed_neural_readout.py",
        "scripts/chromaseed_gaussian.py",
        "scripts/chromaseed_gaussian_runtime.py",
        "scripts/chromaseed_gaussian_numpy.py",
        "scripts/chromaseed_perceptual.py",
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
            complete_fits_including_warmups=378,
            head_complete_fits=252,
            unchanged_complete_fits=108,
            FG_complete_fits=18,
            wall_seconds=time.perf_counter() - started,
            scope="381 actual one-row consumers,20 warmups and3 passes.126 seed17 settings x3 uncached complete fits,one warmup/two measured. Learned representations rebuilt from scratch including all selected-basis epochs; random includes fit normalizers/initialization. Heads include feature moments/weights/metric/solve/export. FG includes fresh width/centers/readout. One CPU thread; imports/I/O/image processing excluded. Representation parameter-state and explicit head design/metric/system byte counts are phase diagnostics, not peak process RAM; random has no iterative representation state and reports null. Initializer temporaries/data/other arrays/objects remain separate.",
        ),
    )
    print("NR RUNTIME PASS:378 complete matching fits", flush=True)


if __name__ == "__main__":
    main()
