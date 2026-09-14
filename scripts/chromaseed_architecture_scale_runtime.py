"""AS actual CPU response and complete selected-bank reconstruction measurements."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_architecture_scale import RATES, SEEDS, VARIANTS, Predictor, fit, predict_torch
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, bank_path, check_map, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as upstream_fit
from chromaseed_long_training_run import ND, NP, context
from chromaseed_neural_prefix_numpy import export_prefix
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from skin_local_search_train import sha, weights_for, write_json


def response(model, x, tokens, expected):
    begin = time.perf_counter()
    consumer = Predictor(model)
    construction = time.perf_counter() - begin
    indices = np.linspace(0, len(x) - 1, min(64, len(x)), dtype=int)
    for i in range(20):
        j = indices[i % len(indices)]
        consumer(x[j], tokens[j])
    timings = []
    maximum = 0.0
    for _ in range(3):
        for j in indices:
            started = time.perf_counter_ns()
            value = consumer(x[j], tokens[j])
            timings.append((time.perf_counter_ns() - started) / 1000)
            np.testing.assert_allclose(value, expected[j], atol=0.002, rtol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(value - expected[j]))))
    return dict(
        median_us=float(np.median(timings)),
        p95_us=float(np.quantile(timings, 0.95)),
        calls=len(timings),
        distinct_rows=len(indices),
        warmups=20,
        constructor_seconds=construction,
        cached_layer_bytes=sum(w.nbytes + b.nbytes for w, b in consumer.layers.values()),
        maximum_native_lab=maximum,
    )


def main():
    assert not (OUT / "verification.json").exists(), "sealed AS; use report verifier"
    setup("cuda")
    started = time.perf_counter()
    lock, selection, results = [
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
    responses, recipes = [], []
    for rec in results["records"]:
        if "imported_we_record" in rec:
            continue
        model = nz(ROOT / rec["model"])
        saved = nz(ROOT / rec["output"])
        ix = saved["row_indices"]
        timing = response(model, data["color"][ix], data["tokens"][ix], saved["predictions"])
        responses.append(dict(role=rec["role"], variant=rec["variant"], seed=rec["seed"], **timing))
    print("AS RESPONSE63 complete", flush=True)
    upstream_count = bank_count = selected_count = whole_bank_count = 0
    for role, choices in selection["roles"].items():
        ix, query, warm_expected, _ = context(data, role)
        prior = js(NP / "selections.json")["roles"][role]["blind4"]
        prefix = prior["policies"]["quality"]["prefix"]
        original = [nz(ND / "selected" / role / f"blind4_s{s}.npz") for s in SEEDS]
        weights = weights_for(data["patient"][ix], data["site"][ix])
        for variant in VARIANTS:
            choice = choices["policies"][variant]
            began = time.perf_counter()
            warm, upstream = [], []
            for si, seed in enumerate(SEEDS):
                raw, info = upstream_fit(
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
                parent = raw[prior["step"]][0]
                exact(parent, original[si])
                model = export_prefix(parent, prefix)
                exact(model, warm_expected[si])
                warm.append(model)
                upstream.append(dict(seed=seed, original_and_prefix_bitwise=True, **info))
                upstream_count += 1
            upstream_seconds = time.perf_counter() - began
            fitted, info = fit(
                data["color"][ix],
                data["tokens"][ix],
                data["target"][ix],
                weights,
                warm,
                variant,
                choice["step"],
                (choice["step"],),
                "cuda",
                "cuda_graph",
            )
            build_seconds = time.perf_counter() - began
            expected_bank = nz(bank_path(role, variant) / f"models_{choice['step']}.npz")
            for slot, model in enumerate(fitted[choice["step"]]):
                exact(model, model_from(expected_bank, str(slot)))
                whole_bank_count += 1
            checks = []
            for si, seed in enumerate(SEEDS):
                rec = next(
                    r
                    for r in results["records"]
                    if r["role"] == role and r["variant"] == variant and r["seed"] == seed
                )
                model = fitted[choice["step"]][2 * si + RATES.index(choice["lr"])]
                exact(model, nz(ROOT / rec["model"]))
                out = predict_torch(model, data["color"][query], data["tokens"][query])
                expected = nz(ROOT / rec["output"])["predictions"]
                np.testing.assert_array_equal(out, expected)
                checks.append(dict(seed=seed, model_and_prediction_bitwise=True))
                selected_count += 1
            bank_count += 1
            recipes.append(
                dict(
                    role=role,
                    variant=variant,
                    step=choice["step"],
                    lr=choice["lr"],
                    upstream_seconds=upstream_seconds,
                    continuation=info,
                    upstream=upstream,
                    build_seconds=build_seconds,
                    validation_inclusive_seconds=time.perf_counter() - began,
                    checks=checks,
                    full_bank_slots=6,
                    full_replays=1,
                    timing_scope="one full three-seed upstream plus six-slot continuation replay; no per-model division",
                )
            )
            print("AS RECIPE", role, variant, choice["step"], round(build_seconds, 3), flush=True)
    assert len(responses) == 63 and sum(r["calls"] for r in responses) == 12096
    assert upstream_count == selected_count == 63 and bank_count == 21 and whole_bank_count == 126
    value = dict(
        passed=True,
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        responses=responses,
        recipes=recipes,
        upstream_fits=upstream_count,
        continuation_banks=bank_count,
        selected_payloads_bitwise=selected_count,
        all_bank_payloads_bitwise=whole_bank_count,
        dependencies={
            "scripts/chromaseed_architecture_scale_runtime.py": sha(
                ROOT / "scripts/chromaseed_architecture_scale_runtime.py"
            )
        },
        seconds=time.perf_counter() - started,
    )
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
        }
    )
    write_json(OUT / "runtime.json", value)
    print("AS RUNTIME PASSED", selected_count, whole_bank_count, flush=True)


if __name__ == "__main__":
    main()
