"""Post-failure NB coverage: measure every guard, never relax or hide failures."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_gate_stability import summaries
from chromaseed_gated import unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_train import save_npz
from chromaseed_neural_blocks_fit import to_prefix
from chromaseed_neural_blocks_run import (
    CHECKPOINTS,
    NP,
    OUT,
    ROOT,
    RUN,
    SEEDS,
    SPECS,
    bank_path,
    check_map,
    load_data,
    outputs,
    settings,
)
from chromaseed_neural_prefix_numpy import capacity, export_prefix
from skin_local_search_train import metrics, roles, sha, write_json


def array_guards(actual, expected):
    assert set(actual) == set(expected)
    exact, passed, maximum = True, True, 0.0
    for key in actual:
        a, b = actual[key], expected[key]
        assert a.shape == b.shape and a.dtype == b.dtype
        exact &= np.array_equal(a, b)
        if a.dtype.kind in "biufc":
            assert np.isfinite(a).all() and np.isfinite(b).all()
            maximum = max(maximum, float(np.max(abs(a.astype(float) - b.astype(float)))))
            passed &= np.allclose(a, b, rtol=2e-6, atol=2e-6)
        else:
            passed &= np.array_equal(a, b)
    return dict(bitwise_equal=bool(exact), weight_guard=bool(passed), max_weight_drift=maximum)


def freeze():
    primary = js(RUN / "source_lock.json")
    check_map({**primary["sources"], **primary["input_sha256"]})
    inputs = {
        (RUN / "source_lock.json").relative_to(ROOT).as_posix(): sha(RUN / "source_lock.json")
    }
    receipts = list((RUN / "banks").glob("*/*/*/receipt.json"))
    assert len(receipts) == 27
    for receiptpath in receipts:
        receipt = js(receiptpath)
        assert receipt["source_lock_sha256"] == sha(RUN / "source_lock.json")
        inputs[receiptpath.relative_to(ROOT).as_posix()] = sha(receiptpath)
        for name, digest in receipt["files"].items():
            path = receiptpath.parent / name
            assert sha(path) == digest
            inputs[path.relative_to(ROOT).as_posix()] = digest
    # Preserve the primary's already-written partial evaluations as well.
    for subdir in ("selected", "evaluated"):
        for path in (RUN / subdir).glob("*/*.npz"):
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    sources = {
        name: sha(ROOT / name)
        for name in (
            "scripts/chromaseed_neural_blocks_diagnose.py",
            "docs/research/chromaseed_neural_blocks_failure_diagnostic.md",
        )
    }
    value = dict(
        primary_source_lock_sha256=sha(RUN / "source_lock.json"),
        primary_session=12062,
        primary_exit_code=1,
        original_equivalence_guards_unchanged=True,
        sources=sources,
        input_sha256=inputs,
        coverage_records=243,
    )
    path = RUN / "diagnostic_source_lock.json"
    if path.exists():
        assert js(path) == value
    else:
        write_json(path, value)
    return sha(path)


def comparison(actual, expected, pred, target, data, query):
    value = array_guards(actual, expected)
    drift = float(np.max(abs(pred - target)))
    error = metrics(pred[0], data["target"][query], data["patient"][query], data["site"][query])[
        "person_mean"
    ]
    reference_error = metrics(
        target[0], data["target"][query], data["patient"][query], data["site"][query]
    )["person_mean"]
    difference = error - reference_error
    value.update(
        max_prediction_drift=drift,
        prediction_guard=drift <= 0.002,
        error_difference=difference,
        error_guard=abs(difference) <= 0.002,
    )
    value["all_guards_pass"] = (
        value["weight_guard"] and value["prediction_guard"] and value["error_guard"]
    )
    return value


def main():
    assert not (OUT / "verification.json").exists(), "sealed NB: report verifier only"
    start = time.perf_counter()
    source = freeze()
    data, records = load_data(), []
    for setting in settings():
        role, family = setting["role"], setting["family"]
        _, held = roles(data["patient"], data["device"])[role]
        query = np.flatnonzero(held)
        for step in CHECKPOINTS:
            full = nz(bank_path(setting, None) / f"raw_{step}.npz")
            for prefix in (None, *setting["prefixes"]):
                raw = nz(bank_path(setting, prefix) / f"raw_{step}.npz")
                mode = "full" if prefix is None else "subset"
                j = SPECS[family][0] if prefix is None else prefix
                for si, seed in enumerate(SEEDS):
                    a, b = unpack(raw, str(si)), unpack(full, str(si))
                    model = export_prefix(a, j) if prefix is None else to_prefix(a)
                    matched = export_prefix(b, j)
                    name = f"{family}_{mode}{j}_t{step}_s{seed}"
                    path = RUN / "diagnosed_models" / role / f"{name}.npz"
                    pp = RUN / "diagnosed_predictions" / role / f"{name}.npz"
                    save_npz(path, model)
                    pred, target = (
                        outputs(model, data["color"][query]),
                        outputs(matched, data["color"][query]),
                    )
                    comparison_info = comparison(model, matched, pred, target, data, query)
                    raw_comparison = array_guards(
                        {"theta": a["theta"]},
                        {"theta": b["theta"] if prefix is None else b["theta"][a["block_indices"]]},
                    )
                    comparison_info["all_guards_pass"] &= raw_comparison["weight_guard"]
                    save_npz(pp, dict(row_indices=query, predictions=pred))
                    transforms, doses = summaries(
                        pred,
                        data["target"][query],
                        data["patient"][query],
                        data["device"][query],
                        None,
                    )
                    np_check = None
                    if step == setting["step"]:
                        parentpath = NP / "selected" / role / f"{family}_j{j}_s{seed}.npz"
                        previous = nz(parentpath)
                        expected = nz(NP / "evaluated" / role / f"{family}_j{j}_s{seed}.npz")[
                            "predictions"
                        ]
                        np_check = dict(
                            parent_name=parentpath.stem,
                            parent_model_sha256=sha(parentpath),
                            **comparison(model, previous, pred, expected, data, query),
                        )
                    records.append(
                        dict(
                            role=role,
                            family=family,
                            name=name,
                            mode=mode,
                            prefix=j,
                            seed=seed,
                            step=step,
                            lr=setting["lr"],
                            selected_step=step == setting["step"],
                            policies=[]
                            if prefix is None or step != setting["step"]
                            else [p for p, value in setting["policies"].items() if value == j],
                            **capacity(model),
                            model_sha256=sha(path),
                            prediction_sha256=sha(pp),
                            archive_bytes=path.stat().st_size,
                            metrics=metrics(
                                pred[0],
                                data["target"][query],
                                data["patient"][query],
                                data["site"][query],
                                data["device"][query],
                            ),
                            transforms=transforms,
                            doses=doses,
                            raw_comparison=raw_comparison,
                            full_control_comparison=comparison_info,
                            np_comparison=np_check,
                        )
                    )
            print("NB DIAGNOSED", role, family, step, flush=True)
    assert len(records) == 243
    value = dict(
        source_lock_sha256=sha(RUN / "source_lock.json"),
        diagnostic_source_lock_sha256=source,
        parent_selection_sha256=sha(NP / "selections.json"),
        primary_exit_code=1,
        equivalence_accepted=False,
        original_guards_unchanged=True,
        records=records,
        seconds=time.perf_counter() - start,
    )
    path = RUN / "diagnostic_results.json"
    assert not path.exists(), "use read-only report verification after measurement"
    write_json(path, value)
    subset = [r for r in records if r["mode"] == "subset"]
    print(
        "NB DIAGNOSIS COMPLETE",
        dict(
            subset=len(subset),
            failed=sum(not r["full_control_comparison"]["all_guards_pass"] for r in subset),
            selected_failed=sum(
                r["selected_step"] and not r["full_control_comparison"]["all_guards_pass"]
                for r in subset
            ),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
