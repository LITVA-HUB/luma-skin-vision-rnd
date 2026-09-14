"""WE consumer timing and full selected six-slot training reconstruction."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as upstream_fit
from chromaseed_long_training_run import ND, NP, context
from chromaseed_neural_prefix_numpy import export_prefix
from chromaseed_refine_train import setup
from chromaseed_widen import SEEDS, predict
from chromaseed_widen_audit import exact
from chromaseed_widen_early_fit import fit
from chromaseed_widen_early_run import CAPS, OUT, ROOT, RUN, check_map, load_data
from chromaseed_widen_runtime import time_response
from skin_local_search_train import sha, weights_for, write_json


def main():
    assert not (OUT / "verification.json").exists(), "sealed WE; report verifier only"
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
    for rec in result["records"]:
        model = nz(RUN / "models" / rec["role"] / f"{rec['name']}.npz")
        saved = nz(RUN / "evaluated" / rec["role"] / f"{rec['name']}.npz")
        ix = saved["row_indices"]
        timing = time_response(
            model, data["color"][ix], data["tokens"][ix], saved["predictions"][0]
        )
        records.append(
            dict(
                role=rec["role"],
                kind=rec["kind"],
                variant=rec["variant"],
                seed=rec["seed"],
                name=rec["name"],
                **timing,
            )
        )
    print("WE RUNTIME90 consumers complete", flush=True)
    recipes = []
    original_count = bank_count = selected_count = bank_payload_count = 0
    for role, entry in selection["roles"].items():
        ix, query, warm_expected, _ = context(data, role)
        prior = js(NP / "selections.json")["roles"][role]["blind4"]
        prefix = prior["policies"]["quality"]["prefix"]
        originals = [nz(ND / "selected" / role / f"blind4_s{s}.npz") for s in SEEDS]
        for variant in CAPS:
            choice = entry["policies"][variant]
            first = 0.0001 if choice["lr"] == 0.001 else choice["lr"]
            chosen = [
                r
                for r in result["records"]
                if r["kind"] == "new" and r["role"] == role and r["variant"] == variant
            ]
            assert len(chosen) == 3
            expected = {r["seed"]: nz(RUN / "models" / role / f"{r['name']}.npz") for r in chosen}
            predictions = {
                r["seed"]: nz(RUN / "evaluated" / role / f"{r['name']}.npz")["predictions"][0]
                for r in chosen
            }
            expected_bank = nz(ROOT / chosen[0]["source_path"])
            repeats = []
            for repetition in range(3):
                began = time.perf_counter()
                weights = weights_for(data["patient"][ix], data["site"][ix])
                warm = []
                upstream = []
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
                    exact(parent, originals[si])
                    model = export_prefix(parent, prefix)
                    exact(model, warm_expected[si])
                    warm.append(model)
                    upstream.append(dict(seed=seed, original_and_prefix_bitwise=True, **info))
                    original_count += 1
                upstream_seconds = time.perf_counter() - began
                fitted, info = fit(
                    data["color"][ix],
                    data["tokens"][ix],
                    data["target"][ix],
                    weights,
                    warm,
                    variant,
                    first,
                    choice["step"],
                    (0, choice["step"]),
                    "cuda",
                    "cuda_graph",
                )
                bank_count += 1
                build_seconds = time.perf_counter() - began
                for slot, m in enumerate(fitted[choice["step"]]):
                    exact(m, model_from(expected_bank, str(slot)))
                    bank_payload_count += 1
                checks = []
                for si, seed in enumerate(SEEDS):
                    m = fitted[choice["step"]][2 * si + (choice["lr"] == 0.001)]
                    exact(m, expected[seed])
                    selected_count += 1
                    drift = float(
                        np.max(
                            abs(
                                predict(m, data["color"][query], data["tokens"][query])
                                - predictions[seed]
                            )
                        )
                    )
                    assert drift <= 2e-8
                    checks.append(dict(seed=seed, bitwise_equal=True, max_prediction_drift=drift))
                repeats.append(
                    dict(
                        repetition=repetition,
                        timing_warmup=repetition == 0,
                        upstream_seconds=upstream_seconds,
                        build_seconds=build_seconds,
                        reconstruction_and_validation_seconds=time.perf_counter() - began,
                        upstream=upstream,
                        continuation=info,
                        checks=checks,
                    )
                )
            record = dict(
                role=role,
                variant=variant,
                step=choice["step"],
                lr=choice["lr"],
                first_rate=first,
                bank_slots=6,
                original_horizon=8192,
                upstream_seeds=list(SEEDS),
                original_steps=prior["step"],
                original_rate=prior["lr"],
                original_prefix=prefix,
                median_build_seconds=float(np.median([r["build_seconds"] for r in repeats[1:]])),
                median_reconstruction_and_validation_seconds=float(
                    np.median([r["reconstruction_and_validation_seconds"] for r in repeats[1:]])
                ),
                repeats=repeats,
            )
            recipes.append(record)
            print(
                "WE RECIPE",
                role,
                variant,
                choice["step"],
                choice["lr"],
                round(record["median_reconstruction_and_validation_seconds"], 4),
                flush=True,
            )
    assert len(records) == 90 and sum(r["calls"] for r in records) == 107820
    assert (
        len(recipes) == 12
        and original_count == selected_count == 108
        and bank_count == 36
        and bank_payload_count == 216
    )
    deps = {
        "scripts/chromaseed_widen_early_runtime.py": sha(
            ROOT / "scripts/chromaseed_widen_early_runtime.py"
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
        upstream_singleton_fits=original_count,
        continuation_banks=bank_count,
        selected_payloads_bitwise=selected_count,
        all_bank_payloads_bitwise=bank_payload_count,
        dependencies=deps,
        seconds=time.perf_counter() - start,
        cost_scope="Warm process, original TRAIN already loaded; three original ND singleton trainings plus NP export and complete six-slot WE continuation to selected step with original8192 horizon, including setup/export and separate validation. No division by slots. Historical WIDE training timings are not freshly paired measurements. CPU responses include token normalization/encoding, exclude image decoding/extraction.",
    )
    write_json(OUT / "runtime.json", value)
    print("WE RUNTIME PASSED108 selected and216 full-bank exports exact", flush=True)


if __name__ == "__main__":
    main()
