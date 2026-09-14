"""LT one-vector response and full, fixed-bank selected recipe construction."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as upstream_fit
from chromaseed_long_training_fit import SEEDS, fit
from chromaseed_long_training_run import (
    ND,
    NP,
    OUT,
    ROOT,
    RUN,
    check_map,
    context,
    load_data,
    slot_index,
)
from chromaseed_neural_prefix_numpy import export_prefix, predict
from chromaseed_neural_prefix_runtime import time_consumer
from chromaseed_refine_train import setup
from skin_local_search_train import sha, weights_for, write_json


def assert_identical(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def main():
    assert not (OUT / "verification.json").exists(), "sealed LT: report verifier only"
    setup("cuda")
    started = time.perf_counter()
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
    for rec in result["records"]:
        role, name = rec["role"], rec["name"]
        model = nz(RUN / "selected" / role / f"{name}.npz")
        saved = nz(RUN / "evaluated" / role / f"{name}.npz")
        measured = time_consumer(
            model, data["color"][saved["row_indices"]], saved["predictions"][0]
        )
        assert measured["cached_array_bytes"] == 5456
        records.append(
            dict(
                role=role,
                name=name,
                seed=rec["seed"],
                variants=rec["variants"],
                step=rec["step"],
                lr=rec["lr"],
                **measured,
            )
        )
    print("LT RUNTIME consumers", len(records), flush=True)
    recipes, upstream_count, bank_count, selected_count = [], 0, 0, 0
    for role, entry in selection["roles"].items():
        ix, query, expected_warm, _ = context(data, role)
        prior = js(NP / "selections.json")["roles"][role]["blind4"]
        j = prior["policies"]["quality"]["prefix"]
        parents = [nz(ND / "selected" / role / f"blind4_s{s}.npz") for s in SEEDS]
        selected = [
            r
            for r in result["records"]
            if r["role"] == role and (r["selected_per_mode"] or r["selected_overall"])
        ]
        expected = {r["name"]: nz(RUN / "selected" / role / f"{r['name']}.npz") for r in selected}
        expected_prediction = {
            r["name"]: nz(RUN / "evaluated" / role / f"{r['name']}.npz")["predictions"][0]
            for r in selected
        }
        checkpoints = sorted(
            {0, *(c["step"] for c in entry["per_mode"].values()), entry["overall"]["step"]}
        )
        steps, repeats = max(checkpoints), []
        for repetition in range(3):
            began = time.perf_counter()
            weights = weights_for(data["patient"][ix], data["site"][ix])
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
                original = raw[prior["step"]][0]
                model = export_prefix(original, j)
                assert_identical(original, parents[si])
                assert_identical(model, expected_warm[si])
                warm.append(model)
                upstream.append(
                    dict(seed=seed, original_bitwise_equal=True, prefix_bitwise_equal=True, **info)
                )
                upstream_count += 1
            upstream_seconds = time.perf_counter() - began
            continuation = None
            if steps:
                models, continuation = fit(
                    data["color"][ix],
                    data["target"][ix],
                    weights,
                    ix,
                    warm,
                    steps,
                    tuple(checkpoints),
                    "cuda",
                    "cuda_graph",
                )
                bank_count += 1
            else:
                models = {0: [warm[si] for si in range(3) for _ in range(6)]}
            build_seconds = time.perf_counter() - began
            checks = []
            for rec in selected:
                si = SEEDS.index(rec["seed"])
                actual = models[rec["step"]][slot_index(si, rec["variants"], rec["lr"])]
                assert_identical(actual, expected[rec["name"]])
                drift = float(
                    np.max(
                        abs(
                            predict(actual, data["color"][query]) - expected_prediction[rec["name"]]
                        )
                    )
                )
                assert drift <= 2e-8
                checks.append(
                    dict(name=rec["name"], bitwise_equal=True, max_prediction_drift=drift)
                )
                selected_count += 1
            full_seconds = time.perf_counter() - began
            repeats.append(
                dict(
                    repetition=repetition,
                    timing_warmup=repetition == 0,
                    upstream_seconds=upstream_seconds,
                    build_seconds=build_seconds,
                    reconstruction_and_validation_seconds=full_seconds,
                    upstream=upstream,
                    continuation=continuation,
                    checks=checks,
                )
            )
        record = dict(
            role=role,
            upstream_seeds=list(SEEDS),
            original_family="blind4",
            original_prefix=j,
            original_rate=prior["lr"],
            original_steps=prior["step"],
            continuation_steps=steps,
            checkpoints=checkpoints,
            bank_slots=18 if steps else 0,
            original_horizon=131072,
            lt_training_selected=steps > 0,
            selected_payload_aliases=len(selected),
            median_build_seconds=float(np.median([r["build_seconds"] for r in repeats[1:]])),
            median_upstream_seconds=float(np.median([r["upstream_seconds"] for r in repeats[1:]])),
            median_reconstruction_and_validation_seconds=float(
                np.median([r["reconstruction_and_validation_seconds"] for r in repeats[1:]])
            ),
            repeats=repeats,
        )
        recipes.append(record)
        print(
            "LT RECIPE",
            role,
            "steps",
            steps,
            "seconds",
            round(record["median_reconstruction_and_validation_seconds"], 4),
            flush=True,
        )
    assert len(records) == 324 and upstream_count == 27 and selected_count == 81
    assert bank_count == 3 * sum(r["lt_training_selected"] for r in recipes)
    assert sum(r["calls"] for r in records) == 388152
    dependencies = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_long_training_runtime",
            "chromaseed_long_training_fit",
            "chromaseed_long_training_run",
            "chromaseed_neural_prefix_runtime",
            "chromaseed_neural_prefix_numpy",
            "chromaseed_local_denoise_fit",
            "chromaseed_refine_train",
            "skin_local_search_train",
        )
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
        upstream_original_and_prefix_bitwise=upstream_count,
        continuation_banks=bank_count,
        selected_payload_reconstructions=selected_count,
        selected_payloads_bitwise=selected_count,
        dependencies=dependencies,
        seconds=time.perf_counter() - started,
        cost_scope="Warm process/CUDA context; original TRAIN already loaded. Three full original ND singleton fits, NP exports and their validation; full fixed18-slot LT bank to maximum selected step with original131072 horizon, all pool/sample/setup/graph/export costs. Separate build and build+final-validation times. First of3 repetitions omitted only from timing. No division by slots, no one-head training claim. CPU is one prepared color36 vector with normalization; face/skin extraction and process/file I/O excluded.",
    )
    write_json(OUT / "runtime.json", value)
    print(
        "LT RUNTIME PASSED",
        dict(upstream_fits=upstream_count, banks=bank_count, bitwise_selected=selected_count),
        flush=True,
    )


if __name__ == "__main__":
    main()
