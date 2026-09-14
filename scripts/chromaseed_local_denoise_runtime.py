"""Actual ND one-row timings and complete single-slot GPU reconstruction costs."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gaussian_audit import actual_consumer
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit
from chromaseed_local_denoise_numpy import Predictor, predict
from chromaseed_local_denoise_train import ROOT, RUN, load_data
from skin_local_search_train import roles, sha, weights_for, write_json

OUT = ROOT / "docs/benchmarks/chromaseed_local_denoise_v1"


def query_time(model, x, expected):
    t = time.perf_counter()
    consumer = Predictor(model) if "theta" in model else actual_consumer(model)
    constructor = time.perf_counter() - t
    for i in range(20):
        consumer(x[i % len(x)])
    timings = []
    maximum = 0.0
    for _ in range(3):
        for row, target in zip(x, expected, strict=True):
            start = time.perf_counter_ns()
            result = consumer(row)
            timings.append((time.perf_counter_ns() - start) / 1000)
            maximum = max(maximum, float(np.max(abs(result - target))))
    assert maximum <= 2e-8, maximum
    cached = getattr(consumer, "cached_array_bytes", None)
    if cached is None:
        # Existing reference consumers own an explicit cached-bytes accessor.
        cached = getattr(consumer, "cached_bytes", None)
        cached = cached() if callable(cached) else cached
    return dict(
        median_us=float(np.median(timings)),
        p95_us=float(np.percentile(timings, 95)),
        calls=len(timings),
        warmups=20,
        constructor_seconds=constructor,
        cached_array_bytes=cached,
        maximum_prediction_drift=maximum,
    )


def main():
    started = time.perf_counter()
    audit = js(OUT / "audit.json")
    assert audit["passed"] and audit["results_sha256"] == sha(RUN / "results.json")
    lock = js(RUN / "source_lock.json")
    for path, digest in {
        **lock["sources"],
        **lock["input_sha256"],
        **audit["artifact_sha256"],
        **audit["dependencies"],
    }.items():
        assert sha(ROOT / path) == digest, path
    data = load_data()
    result = js(RUN / "results.json")
    records = []
    full_fits = []
    bitwise = 0
    for rec in result["records"]:
        role, name = rec["role"], rec["name"]
        model = nz(RUN / "selected" / role / f"{name}.npz")
        prediction = nz(RUN / "evaluated" / role / f"{name}.npz")
        rows = prediction["row_indices"]
        output = prediction["stages"][0, :, -1]
        timing = query_time(model, data["color"][rows], output)
        records.append(dict(role=role, name=name, family=rec["family"], seed=rec["seed"], **timing))
    for rec in result["records"]:
        if rec["origin"] != "ND_fit" or rec["seed"] != 17:
            continue
        role, name = rec["role"], rec["name"]
        mask, held = roles(data["patient"], data["device"])[role]
        expected = nz(RUN / "selected" / role / f"{name}.npz")
        expected_predictions = nz(RUN / "evaluated" / role / f"{name}.npz")["stages"][0]
        w = weights_for(data["patient"][mask], data["site"][mask])
        repeats = []
        for repetition in range(3):
            models, info = fit(
                data["color"][mask],
                data["target"][mask],
                w,
                rec["family"],
                [(17, rec["lr"])],
                rec["step"],
                (rec["step"],),
                "cuda",
                "cuda_graph",
            )
            actual = models[rec["step"]][0]
            identical = True
            for key in expected:
                identical &= np.array_equal(actual[key], expected[key])
                if expected[key].dtype.kind in "biufc":
                    np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=2e-6)
                else:
                    np.testing.assert_array_equal(actual[key], expected[key])
            output = predict(actual, data["color"][held])
            drift = float(np.max(abs(output - expected_predictions)))
            assert drift <= 0.002, drift
            bitwise += int(identical)
            repeats.append(
                dict(
                    repetition=repetition,
                    warmup=repetition == 0,
                    bitwise_equal=bool(identical),
                    maximum_native_lab_drift=drift,
                    **info,
                )
            )
        full_fits.append(
            dict(
                role=role,
                name=name,
                family=rec["family"],
                seed=17,
                step=rec["step"],
                lr=rec["lr"],
                full_fit_seconds=float(np.median([r["full_bank_seconds"] for r in repeats[1:]])),
                repeats=repeats,
            )
        )
        print(
            "TIMED full fit",
            role,
            rec["family"],
            round(full_fits[-1]["full_fit_seconds"] * 1000, 3),
            "ms",
            flush=True,
        )
    assert len(records) == 63 and len(full_fits) == 15
    dependencies = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_local_denoise_runtime",
            "chromaseed_local_denoise_fit",
            "chromaseed_local_denoise_numpy",
            "chromaseed_gaussian_audit",
            "chromaseed_refine",
            "chromaseed_refine_train",
        )
    }
    value = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        records=records,
        full_fit_records=full_fits,
        reconstructions=45,
        bitwise_equal_reconstructions=bitwise,
        dependencies=dependencies,
        seconds=time.perf_counter() - started,
        cost_scope="Complete single-slot construction; first timing discarded; reference fit timings not remeasured",
    )
    write_json(OUT / "runtime.json", value)
    print("ND RUNTIME COMPLETE", dict(reconstructions=45, bitwise_equal=bitwise), flush=True)


if __name__ == "__main__":
    main()
