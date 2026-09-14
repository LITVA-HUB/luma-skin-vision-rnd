"""NP consumer speed and full original ND training plus exact prefix export."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gaussian_audit import actual_consumer
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit
from chromaseed_local_denoise_numpy import Predictor as NDPredictor
from chromaseed_neural_prefix_numpy import Predictor, export_prefix, predict
from chromaseed_neural_prefix_run import ND, OUT, ROOT, RUN, check_map, load_data
from skin_local_search_train import roles, sha, weights_for, write_json


def time_consumer(model, x, expected):
    start = time.perf_counter()
    if "original_k" in model:
        consumer = Predictor(model)
    elif "theta" in model:
        consumer = NDPredictor(model)
    else:
        consumer = actual_consumer(model)
    constructor = time.perf_counter() - start
    for i in range(20):
        consumer(x[i % len(x)])
    timing, maximum = [], 0.0
    for _ in range(3):
        for row, target in zip(x, expected, strict=True):
            start = time.perf_counter_ns()
            result = consumer(row)
            timing.append((time.perf_counter_ns() - start) / 1000)
            maximum = max(maximum, float(np.max(abs(result - target))))
    assert maximum <= 2e-8
    cached = getattr(consumer, "cached_array_bytes", None)
    if cached is None:
        cached = getattr(consumer, "cached_bytes", None)
        cached = cached() if callable(cached) else cached
    return dict(
        median_us=float(np.median(timing)),
        p95_us=float(np.percentile(timing, 95)),
        calls=len(timing),
        warmups=20,
        constructor_seconds=constructor,
        cached_array_bytes=cached,
        maximum_prediction_drift=maximum,
    )


def main():
    assert not (OUT / "verification.json").exists(), (
        "sealed measurement: use report read-only verifier"
    )
    start = time.perf_counter()
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    audit = js(OUT / "audit.json")
    assert audit["passed"] and audit["results_sha256"] == sha(RUN / "results.json")
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
        }
    )
    data = load_data()
    records, controls, fits = [], [], []
    for rec in result["records"]:
        role, name = rec["role"], rec["name"]
        model = nz(RUN / "selected" / role / f"{name}.npz")
        saved = nz(RUN / "evaluated" / role / f"{name}.npz")
        x, expected = data["color"][saved["row_indices"]], saved["predictions"][0]
        timing = time_consumer(model, x, expected)
        records.append(
            dict(
                role=role,
                name=name,
                family=rec["family"],
                seed=rec["seed"],
                prefix=rec["prefix"],
                **timing,
            )
        )
        if rec["full_prefix"]:
            original = nz(ND / "selected" / role / f"{rec['parent_name']}.npz")
            controls.append(
                dict(
                    role=role,
                    name=rec["parent_name"],
                    prefix_name=name,
                    family=rec["family"],
                    seed=rec["seed"],
                    **time_consumer(original, x, expected),
                )
            )
    chosen = [
        r
        for r in result["records"]
        if r["prefix"] is not None and r["seed"] == 17 and (r["policies"] or r["full_prefix"])
    ]
    bitwise, original_bitwise = 0, 0
    for rec in chosen:
        role, name = rec["role"], rec["name"]
        mask, held = roles(data["patient"], data["device"])[role]
        expected = nz(RUN / "selected" / role / f"{name}.npz")
        parent = nz(ND / "selected" / role / f"{rec['parent_name']}.npz")
        expected_output = nz(RUN / "evaluated" / role / f"{name}.npz")["predictions"][0]
        repeats = []
        for repetition in range(3):
            began = time.perf_counter()
            w = weights_for(data["patient"][mask], data["site"][mask])
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
            original = models[rec["step"]][0]
            export_start = time.perf_counter()
            actual = export_prefix(original, rec["prefix"])
            export_seconds = time.perf_counter() - export_start
            full_seconds = time.perf_counter() - began
            assert set(original) == set(parent) and set(actual) == set(expected)
            for key in original:
                np.testing.assert_array_equal(original[key], parent[key])
            original_bitwise += 1
            for key in actual:
                np.testing.assert_array_equal(actual[key], expected[key])
            bitwise += 1
            drift = float(np.max(abs(predict(actual, data["color"][held]) - expected_output)))
            assert drift <= 2e-8
            repeats.append(
                dict(
                    repetition=repetition,
                    warmup=repetition == 0,
                    original_bitwise_equal=True,
                    prefix_bitwise_equal=True,
                    max_prediction_drift=drift,
                    reconstruction_seconds=full_seconds,
                    prefix_export_seconds=export_seconds,
                    original_fit_receipt=info,
                )
            )
        fits.append(
            dict(
                role=role,
                name=name,
                family=rec["family"],
                prefix=rec["prefix"],
                seed=17,
                full_prefix=rec["full_prefix"],
                policies=rec["policies"],
                step=rec["step"],
                lr=rec["lr"],
                full_fit_seconds=float(
                    np.median([r["reconstruction_seconds"] for r in repeats[1:]])
                ),
                original_fit_seconds=float(
                    np.median([r["original_fit_receipt"]["full_bank_seconds"] for r in repeats[1:]])
                ),
                prefix_export_seconds=float(
                    np.median([r["prefix_export_seconds"] for r in repeats[1:]])
                ),
                repeats=repeats,
            )
        )
        print(
            "NP FULL FIT",
            role,
            name,
            round(fits[-1]["full_fit_seconds"] * 1000, 3),
            "ms",
            flush=True,
        )
    expected_fits = sum(
        len(
            {
                entry["policies"]["quality"]["prefix"],
                entry["policies"]["compact"]["prefix"],
                len(entry["candidates"]),
            }
        )
        for family in selection["roles"].values()
        for entry in family.values()
    )
    assert len(records) == 153 and len(controls) == 45 and len(fits) == expected_fits
    assert bitwise == original_bitwise == expected_fits * 3
    dependencies = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_neural_prefix_runtime",
            "chromaseed_neural_prefix_numpy",
            "chromaseed_local_denoise_fit",
            "chromaseed_local_denoise_numpy",
            "chromaseed_refine_train",
            "chromaseed_refine",
            "chromaseed_gaussian_audit",
        )
    }
    value = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        records=records,
        original_ND_controls=controls,
        full_fit_records=fits,
        unique_full_fit_settings=expected_fits,
        reconstructions=bitwise,
        bitwise_equal_original_reconstructions=original_bitwise,
        bitwise_equal_prefix_reconstructions=bitwise,
        dependencies=dependencies,
        seconds=time.perf_counter() - start,
        cost_scope="Full original single-model ND construction + prefix export; first of3 warmups excluded only from time. CPU response is prepared features; all original training charged. FG fits not remeasured.",
    )
    write_json(OUT / "runtime.json", value)
    print("NP RUNTIME COMPLETE", dict(settings=expected_fits, reconstructions=bitwise), flush=True)


if __name__ == "__main__":
    main()
