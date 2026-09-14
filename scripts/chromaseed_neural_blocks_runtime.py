"""Complete paired NB fit costs; correctness failures remain explicit."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as full_fit
from chromaseed_neural_blocks_diagnose import array_guards
from chromaseed_neural_blocks_fit import fit as subset_fit
from chromaseed_neural_blocks_fit import to_prefix
from chromaseed_neural_blocks_run import NP, OUT, ROOT, RUN, check_map, load_data, outputs, settings
from chromaseed_neural_prefix_numpy import export_prefix
from skin_local_search_train import metrics, roles, sha, weights_for, write_json


def checked(model, expected, prediction, reference, data, query):
    result = array_guards(model, expected)
    delta = float(np.max(abs(prediction - reference)))
    actual_error = metrics(
        prediction[0], data["target"][query], data["patient"][query], data["site"][query]
    )["person_mean"]
    expected_error = metrics(
        reference[0], data["target"][query], data["patient"][query], data["site"][query]
    )["person_mean"]
    result.update(
        max_prediction_drift=delta,
        prediction_guard=delta <= 0.002,
        error_difference=actual_error - expected_error,
        error_guard=abs(actual_error - expected_error) <= 0.002,
    )
    result["all_guards_pass"] = (
        result["weight_guard"] and result["prediction_guard"] and result["error_guard"]
    )
    return result


def main():
    assert not (OUT / "verification.json").exists()
    start = time.perf_counter()
    lock, diag, audit = (
        js(RUN / "source_lock.json"),
        js(RUN / "diagnostic_source_lock.json"),
        js(OUT / "audit.json"),
    )
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **diag["sources"],
            **diag["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
        }
    )
    assert audit["passed"] and not audit["equivalence_accepted"]
    data, records = load_data(), []
    calls = 0
    for setting in settings():
        role, family = setting["role"], setting["family"]
        mask, held = roles(data["patient"], data["device"])[role]
        query = np.flatnonzero(held)
        for j in setting["prefixes"]:
            previous = nz(NP / "selected" / role / f"{family}_j{j}_s17.npz")
            expected = nz(NP / "evaluated" / role / f"{family}_j{j}_s17.npz")["predictions"]
            repeats = []
            for repetition in range(3):
                pair, models, predictions = {}, {}, {}
                order = ("full", "subset") if repetition % 2 == 0 else ("subset", "full")
                for mode in order:
                    began = time.perf_counter()
                    w = weights_for(data["patient"][mask], data["site"][mask])
                    args = (
                        data["color"][mask],
                        data["target"][mask],
                        w,
                        family,
                        [(17, setting["lr"])],
                    )
                    if mode == "full":
                        raw, info = full_fit(
                            *args, setting["step"], (setting["step"],), "cuda", "cuda_graph"
                        )
                        model = export_prefix(raw[setting["step"]][0], j)
                    else:
                        raw, info = subset_fit(
                            *args, j, setting["step"], (setting["step"],), "cuda", "cuda_graph"
                        )
                        model = to_prefix(raw[setting["step"]][0])
                    elapsed = time.perf_counter() - began
                    prediction = outputs(model, data["color"][query])
                    pair[mode] = dict(
                        complete_seconds=elapsed,
                        fit_receipt=info,
                        np_comparison=checked(model, previous, prediction, expected, data, query),
                    )
                    models[mode], predictions[mode] = model, prediction
                    calls += 1
                repeats.append(
                    dict(
                        repetition=repetition,
                        warmup=repetition == 0,
                        order=list(order),
                        full=pair["full"],
                        subset=pair["subset"],
                        paired_comparison=checked(
                            models["subset"],
                            models["full"],
                            predictions["subset"],
                            predictions["full"],
                            data,
                            query,
                        ),
                    )
                )
            full_seconds = float(np.median([r["full"]["complete_seconds"] for r in repeats[1:]]))
            subset_seconds = float(
                np.median([r["subset"]["complete_seconds"] for r in repeats[1:]])
            )
            records.append(
                dict(
                    role=role,
                    family=family,
                    prefix=j,
                    step=setting["step"],
                    lr=setting["lr"],
                    seed=17,
                    full_seconds=full_seconds,
                    subset_seconds=subset_seconds,
                    speedup=full_seconds / subset_seconds,
                    policies=[p for p, value in setting["policies"].items() if value == j],
                    repeats=repeats,
                )
            )
            print(
                "NB PAIRED FIT",
                role,
                family,
                j,
                round(full_seconds * 1000, 3),
                "->",
                round(subset_seconds * 1000, 3),
                "ms",
                flush=True,
            )
    assert calls == 108 and len(records) == 18
    deps = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_neural_blocks_runtime",
            "chromaseed_neural_blocks_diagnose",
            "chromaseed_neural_blocks_fit",
            "chromaseed_local_denoise_fit",
            "chromaseed_neural_prefix_numpy",
        )
    }
    value = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        diagnostic_results_sha256=sha(RUN / "diagnostic_results.json"),
        audit_sha256=sha(OUT / "audit.json"),
        records=records,
        reconstruction_calls=calls,
        dependencies=deps,
        seconds=time.perf_counter() - start,
        equivalence_accepted=False,
        cost_scope="108 complete standalone construction calls,18 settings x3 pairs x2 modes; first pair excluded only for time; all correctness flags retained",
    )
    write_json(OUT / "runtime.json", value)
    print("NB RUNTIME COMPLETE", calls, flush=True)


if __name__ == "__main__":
    main()
