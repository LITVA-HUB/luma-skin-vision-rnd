"""Actual one-example response and complete warm-training plus capacity continuation."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as upstream_fit
from chromaseed_long_training_run import ND, NP, context
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
from chromaseed_neural_prefix_numpy import export_prefix
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_refine_train import setup
from chromaseed_widen import RATES, SEEDS, SPECS, Predictor, fit, predict
from chromaseed_widen_audit import exact
from chromaseed_widen_run import OUT, ROOT, RUN, bank_path, check_map, load_data
from skin_local_search_train import sha, weights_for, write_json


def time_response(model, x, t, expected):
    start = time.perf_counter()
    base = "variant" not in model
    consumer = BasePredictor(model) if base else Predictor(model)
    constructor = time.perf_counter() - start
    for i in range(20):
        consumer(x[i % len(x)]) if base else consumer(x[i % len(x)], t[i % len(x)])
    times = []
    maximum = 0.0
    for _ in range(3):
        for i in range(len(x)):
            begin = time.perf_counter_ns()
            out = consumer(x[i]) if base else consumer(x[i], t[i])
            times.append((time.perf_counter_ns() - begin) / 1000)
            maximum = max(maximum, float(np.max(abs(out - expected[i]))))
    assert maximum <= 2e-8
    return dict(
        median_us=float(np.median(times)),
        p95_us=float(np.percentile(times, 95)),
        calls=len(times),
        warmups=20,
        constructor_seconds=constructor,
        cached_array_bytes=consumer.cached_array_bytes,
        maximum_prediction_drift=maximum,
    )


def main():
    assert not (OUT / "verification.json").exists(), "sealed runtime; use read-only verifier"
    setup("cuda")
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
    records = []
    selected = [r for r in result["records"] if r["stress_evaluated"]]
    for rec in selected:
        m = nz(RUN / "models" / rec["role"] / f"{rec['name']}.npz")
        saved = nz(RUN / "evaluated" / rec["role"] / f"{rec['name']}.npz")
        ix = saved["row_indices"]
        timing = time_response(m, data["color"][ix], data["tokens"][ix], saved["predictions"][0])
        records.append(
            dict(
                role=rec["role"],
                name=rec["name"],
                variant=rec["variant"],
                seed=rec["seed"],
                **timing,
            )
        )
    print("WIDE RUNTIME54 consumers complete", flush=True)
    recipes = []
    upstream_count = bank_count = selected_count = bank_payload_count = 0
    for role, entry in selection["roles"].items():
        ix, query, expected_warm, _ = context(data, role)
        prior = js(NP / "selections.json")["roles"][role]["blind4"]
        j = prior["policies"]["quality"]["prefix"]
        parents = [nz(ND / "selected" / role / f"blind4_s{s}.npz") for s in SEEDS]
        for variant in (*SPECS, "np"):
            chosen = [r for r in selected if r["role"] == role and r["variant"] == variant]
            assert len(chosen) == 3
            expected = {r["seed"]: nz(RUN / "models" / role / f"{r['name']}.npz") for r in chosen}
            output = {
                r["seed"]: nz(RUN / "evaluated" / role / f"{r['name']}.npz")["predictions"][0]
                for r in chosen
            }
            steps = chosen[0]["step"]
            lr = chosen[0]["lr"]
            expected_bank = (
                None if variant == "np" else nz(bank_path(role, variant) / f"models_{steps}.npz")
            )
            repeats = []
            for repetition in range(3):
                began = time.perf_counter()
                weights = weights_for(data["patient"][ix], data["site"][ix])
                warm = []
                upstream = []
                for si, seed in enumerate(SEEDS):
                    raw, receipt = upstream_fit(
                        data["color"][ix],
                        data["target"][ix],
                        weights,
                        "blind4",
                        [(seed, prior["lr"])],
                        prior["step"],
                        (prior["step"],),
                        "cuda",
                        "cuda_graph",
                    )
                    original = raw[prior["step"]][0]
                    exact(original, parents[si])
                    m = export_prefix(original, j)
                    exact(m, expected_warm[si])
                    warm.append(m)
                    upstream.append(
                        dict(
                            seed=seed,
                            original_bitwise_equal=True,
                            prefix_bitwise_equal=True,
                            **receipt,
                        )
                    )
                    upstream_count += 1
                upstream_seconds = time.perf_counter() - began
                continuation = None
                if variant == "np":
                    models = {s: m for s, m in zip(SEEDS, warm, strict=True)}
                else:
                    fitted, continuation = fit(
                        data["color"][ix],
                        data["tokens"][ix],
                        data["target"][ix],
                        weights,
                        warm,
                        variant,
                        steps,
                        (0, steps),
                        "cuda",
                        "cuda_graph",
                    )
                    bank_count += 1
                    models = {
                        seed: fitted[steps][2 * si + RATES.index(lr)]
                        for si, seed in enumerate(SEEDS)
                    }
                build_seconds = time.perf_counter() - began
                if variant != "np":
                    for slot in range(6):
                        exact(fitted[steps][slot], model_from(expected_bank, str(slot)))
                        bank_payload_count += 1
                checks = []
                for seed in SEEDS:
                    exact(models[seed], expected[seed])
                    actual = (
                        base_predict(models[seed], data["color"][query])
                        if variant == "np"
                        else predict(models[seed], data["color"][query], data["tokens"][query])
                    )
                    drift = float(np.max(abs(actual - output[seed])))
                    assert drift <= 2e-8
                    checks.append(dict(seed=seed, bitwise_equal=True, max_prediction_drift=drift))
                    selected_count += 1
                repeats.append(
                    dict(
                        repetition=repetition,
                        timing_warmup=repetition == 0,
                        upstream_seconds=upstream_seconds,
                        build_seconds=build_seconds,
                        reconstruction_and_validation_seconds=time.perf_counter() - began,
                        upstream=upstream,
                        continuation=continuation,
                        checks=checks,
                    )
                )
            rec = dict(
                role=role,
                variant=variant,
                steps=steps,
                lr=lr,
                original_horizon=8192,
                bank_slots=0 if variant == "np" else 6,
                upstream_seeds=list(SEEDS),
                original_rate=prior["lr"],
                original_steps=prior["step"],
                original_prefix=j,
                median_build_seconds=float(np.median([v["build_seconds"] for v in repeats[1:]])),
                median_reconstruction_and_validation_seconds=float(
                    np.median([v["reconstruction_and_validation_seconds"] for v in repeats[1:]])
                ),
                repeats=repeats,
            )
            recipes.append(rec)
            print(
                "WIDE RECIPE",
                role,
                variant,
                steps,
                round(rec["median_reconstruction_and_validation_seconds"], 4),
                flush=True,
            )
    assert (
        len(records) == 54
        and len(recipes) == 18
        and upstream_count == selected_count == 162
        and bank_count == 45
        and bank_payload_count == 270
    )
    assert sum(r["calls"] for r in records) == 64692
    dependencies = {
        "scripts/chromaseed_widen_runtime.py": sha(ROOT / "scripts/chromaseed_widen_runtime.py")
    }
    value = dict(
        passed=True,
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        records=records,
        recipes=recipes,
        upstream_singleton_fits=upstream_count,
        continuation_banks=bank_count,
        selected_payloads_bitwise=selected_count,
        all_bank_payloads_bitwise=bank_payload_count,
        dependencies=dependencies,
        seconds=time.perf_counter() - start,
        cost_scope="Three full original ND singleton fits and NP warm exports, whole fixed six-model capacity bank to selected step with original8192 horizon, setup/sampling/graph/export and validation. No division by slots. Warm process; source TRAIN already loaded. CPU response includes local token normalization and encoding, excludes image decoding and feature extraction.",
    )
    write_json(OUT / "runtime.json", value)
    print("WIDE RUNTIME PASSED162 selected and270 full-bank exports exact", flush=True)


if __name__ == "__main__":
    main()
