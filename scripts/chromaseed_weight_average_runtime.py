"""WA consumer response and honest complete selected-parent reconstruction."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as original_fit
from chromaseed_long_training_fit import fit as continuation_fit
from chromaseed_long_training_run import ND, NP, context, slot_index
from chromaseed_neural_prefix_numpy import export_prefix, predict
from chromaseed_neural_prefix_runtime import time_consumer
from chromaseed_refine_train import setup
from chromaseed_weight_average import average
from chromaseed_weight_average_run import OUT, ROOT, RUN, SEEDS, check_map, load_data, load_parents
from skin_local_search_train import sha, weights_for, write_json


def identical(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def main():
    assert not (OUT / "verification.json").exists(), "sealed WA: report verifier only"
    setup("cuda")
    began = time.perf_counter()
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
    for r in result["records"]:
        p = nz(RUN / "evaluated" / r["role"] / f"{r['name']}.npz")
        m = nz(RUN / "selected" / r["role"] / f"{r['name']}.npz")
        t = time_consumer(m, data["color"][p["row_indices"]], p["predictions"][0])
        assert t["cached_array_bytes"] == 5456
        records.append(
            dict(role=r["role"], name=r["name"], recipe_name=r["recipe_name"], seed=r["seed"], **t)
        )
    print("WA TIMED consumers", len(records), flush=True)
    upstream_count, bank_count, parent_checks, export_checks = 0, 0, 0, 0
    builds = []
    for role in selection["roles"]:
        rr = [r for r in result["records"] if r["role"] == role]
        checkpoints = sorted({0, *(s for r in rr for s in r["steps"])})
        steps = max(checkpoints)
        ix, query, warm_reference, _ = context(data, role)
        prior = js(NP / "selections.json")["roles"][role]["blind4"]
        prefix = prior["policies"]["quality"]["prefix"]
        expected_original = [nz(ND / "selected" / role / f"blind4_s{s}.npz") for s in SEEDS]
        parent = load_parents(role)
        expected = {r["name"]: nz(RUN / "selected" / role / f"{r['name']}.npz") for r in rr}
        expected_pred = {
            r["name"]: nz(RUN / "evaluated" / role / f"{r['name']}.npz")["predictions"][0]
            for r in rr
        }
        repeats = []
        for repeat in range(3):
            start = time.perf_counter()
            weights = weights_for(data["patient"][ix], data["site"][ix])
            warm, upstream = [], []
            for si, seed in enumerate(SEEDS):
                mm, info = original_fit(
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
                original = mm[prior["step"]][0]
                model = export_prefix(original, prefix)
                identical(original, expected_original[si])
                identical(model, warm_reference[si])
                upstream.append(dict(seed=seed, bitwise_original_and_prefix=True, **info))
                warm.append(model)
                upstream_count += 1
            upstream_seconds = time.perf_counter() - start
            info = None
            if steps:
                mm, info = continuation_fit(
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
                mm = {0: [warm[si] for si in range(3) for _ in range(6)]}
            all_parent_checks = 0
            for step in checkpoints:
                for slot, m in enumerate(mm[step]):
                    identical(m, model_from(parent[step], str(slot)))
                    parent_checks += 1
                    all_parent_checks += 1
            validations = []
            average_seconds = []
            for r in rr:
                si = SEEDS.index(r["seed"])
                slot = slot_index(si, r["variants"], r["lr"])
                parts = [mm[s][slot] for s in r["steps"]]
                provenance = [
                    dict(
                        context=f"{role}/final",
                        seed=r["seed"],
                        variants=r["variants"],
                        lr=r["lr"],
                        step=s,
                    )
                    for s in r["steps"]
                ]
                tick = time.perf_counter()
                actual = average(parts, provenance)
                avg_seconds = time.perf_counter() - tick
                average_seconds.append(
                    dict(name=r["name"], seconds=avg_seconds, components=len(parts))
                )
                identical(actual, expected[r["name"]])
                drift = float(
                    np.max(abs(predict(actual, data["color"][query]) - expected_pred[r["name"]]))
                )
                assert drift <= 2e-8
                validations.append(dict(name=r["name"], bitwise=True, max_prediction_drift=drift))
                export_checks += 1
            repeats.append(
                dict(
                    repetition=repeat,
                    timing_warmup=repeat == 0,
                    full_construction_and_validation_seconds=time.perf_counter() - start,
                    upstream_seconds=upstream_seconds,
                    upstream=upstream,
                    continuation=info,
                    exact_parent_checkpoint_payloads=all_parent_checks,
                    averaging=average_seconds,
                    checks=validations,
                )
            )
            print(
                "WA REBUILD",
                role,
                repeat,
                "steps",
                steps,
                "seconds",
                round(repeats[-1]["full_construction_and_validation_seconds"], 3),
                flush=True,
            )
        builds.append(
            dict(
                role=role,
                steps=steps,
                checkpoints=checkpoints,
                slots=18 if steps else 0,
                original_horizon=131072,
                selected_control_payloads=len(rr),
                upstream_seeds=list(SEEDS),
                original_rate=prior["lr"],
                original_steps=prior["step"],
                prefix=prefix,
                full_construction_seconds=float(
                    np.median([r["full_construction_and_validation_seconds"] for r in repeats[1:]])
                ),
                pure_mean_median_us=float(
                    np.median(
                        [
                            a["seconds"] * 1e6
                            for r in repeats[1:]
                            for a in r["averaging"]
                            if a["components"] > 1
                        ]
                    )
                ),
                repeats=repeats,
            )
        )
    assert upstream_count == 27 and bank_count == 3 * sum(b["steps"] > 0 for b in builds)
    assert export_checks == 3 * result["final_models"] and len(records) == result["final_models"]
    assert parent_checks == sum(3 * 18 * len(b["checkpoints"]) for b in builds)
    deps = {
        f"scripts/{n}.py": sha(ROOT / f"scripts/{n}.py")
        for n in (
            "chromaseed_weight_average_runtime",
            "chromaseed_weight_average",
            "chromaseed_weight_average_run",
            "chromaseed_local_denoise_fit",
            "chromaseed_long_training_fit",
            "chromaseed_neural_prefix_runtime",
            "chromaseed_neural_prefix_numpy",
            "chromaseed_refine_train",
        )
    }
    value = dict(
        passed=True,
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        records=records,
        recipes=builds,
        upstream_fits=upstream_count,
        continuation_banks=bank_count,
        exact_parent_checkpoint_payloads=parent_checks,
        exact_final_payloads=export_checks,
        dependencies=deps,
        seconds=time.perf_counter() - began,
        cost_scope="Warm process/CUDA context and loaded original TRAIN. Three full ND singleton fits/NP exports, complete fixed18-slot LT bank at original131072 horizon through maximum required step, all parent checkpoint verification, weight means and exported output checks. First of3 omitted only from time. Entire recipe cost, never bank divided by18 or single-head training. Pure averaging is separately timed and not full training. CPU latency is one prepared color36 vector including normalization, excluding image/face/skin extraction and process/file I/O.",
    )
    write_json(OUT / "runtime.json", value)
    print(
        "WA RUNTIME PASSED",
        dict(
            upstream=upstream_count,
            banks=bank_count,
            parent_exact=parent_checks,
            export_exact=export_checks,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
