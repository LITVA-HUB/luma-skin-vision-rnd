"""Independent NB diagnostic audit; audit validity is not equivalence success."""

from __future__ import annotations

import time

import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, direct, full_metrics, verify_normalizers
from chromaseed_neural_blocks_run import (
    NP,
    OUT,
    ROOT,
    RUN,
    SPECS,
    bank_path,
    check_map,
    load_data,
    settings,
)
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_neural_prefix_numpy import Predictor
from skin_local_search_train import roles, sha, write_json


def guards(a, b):
    assert set(a) == set(b)
    exact, passed, maximum = True, True, 0.0
    for key in a:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        same = np.array_equal(a[key], b[key])
        exact &= same
        if a[key].dtype.kind in "biufc":
            delta = abs(a[key].astype(float) - b[key].astype(float))
            maximum = max(maximum, float(delta.max()))
            passed &= bool(np.all(delta <= 2e-6 + 2e-6 * abs(b[key].astype(float))))
        else:
            passed &= same
    return dict(bitwise_equal=bool(exact), weight_guard=bool(passed), max_weight_drift=maximum)


def sparse_parent(raw):
    family, j = str(raw["family"]), int(raw["prefix"])
    k, d, h = SPECS[family]
    ids = [j - 1] if family == "blind4" else list(range(j))
    np.testing.assert_array_equal(raw["block_indices"], ids)
    assert int(raw["original_k"]) == k and raw["theta"].shape == (
        len(ids),
        (d + 1) * h + (h + 1) * 3,
    )
    value = {key: raw[key] for key in ("family", "x_mean", "x_std", "y_mean", "y_std")}
    value["theta"] = np.zeros((k, raw["theta"].shape[1]), np.float32)
    value["theta"][ids] = raw["theta"]
    return value


def compared(a, b, pred, target, data, query):
    value = guards(a, b)
    drift = float(np.max(abs(pred - target)))
    difference = (
        full_metrics(pred[0], data, query)["person_mean"]
        - full_metrics(target[0], data, query)["person_mean"]
    )
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
    assert not (OUT / "verification.json").exists(), "sealed measurements"
    start = time.perf_counter()
    lock, diag, result = [
        js(RUN / n)
        for n in ("source_lock.json", "diagnostic_source_lock.json", "diagnostic_results.json")
    ]
    check_map(
        {**lock["sources"], **lock["input_sha256"], **diag["sources"], **diag["input_sha256"]}
    )
    assert result["source_lock_sha256"] == sha(RUN / "source_lock.json")
    assert result["diagnostic_source_lock_sha256"] == sha(RUN / "diagnostic_source_lock.json")
    assert result["primary_exit_code"] == 1 and not result["equivalence_accepted"]
    data, artifacts, cache = load_data(), {}, {}
    config = {(s["role"], s["family"]): s for s in settings()}
    counts = dict(
        banks=0,
        models=0,
        subset_models=0,
        subset_bitwise=0,
        subset_guard_pass=0,
        selected_subset=0,
        selected_subset_guard_pass=0,
        np_checks=0,
        np_guard_pass=0,
        prediction_rows=0,
        consumer_calls=0,
        transforms=0,
        doses=0,
    )
    maximum = 0.0
    for path in (RUN / "banks").glob("*/*/*/receipt.json"):
        receipt = js(path)
        setting = config[(receipt["role"], receipt["family"])]
        assert receipt["steps"] == 8192 and receipt["checkpoints"] == [512, 2048, 8192]
        assert receipt["engine"] == "cuda_graph" and receipt["device"] == "cuda"
        assert receipt["slots"] == [dict(seed=s, lr=setting["lr"]) for s in (17, 29, 43)]
        mask, _ = roles(data["patient"], data["device"])[receipt["role"]]
        np.testing.assert_array_equal(
            nz(path.parent / "rows.npz")["fit_rows"], np.flatnonzero(mask)
        )
        for name, digest in receipt["files"].items():
            assert sha(path.parent / name) == digest
            artifacts[(path.parent / name).relative_to(ROOT).as_posix()] = digest
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
        counts["banks"] += 1
    for rec in result["records"]:
        role, family, j, step = rec["role"], rec["family"], rec["prefix"], rec["step"]
        setting = config[(role, family)]
        si = (17, 29, 43).index(rec["seed"])
        is_subset = rec["mode"] == "subset"
        mask, held = roles(data["patient"], data["device"])[role]
        query = np.flatnonzero(held)
        parent = model_from(nz(bank_path(setting, None) / f"raw_{step}.npz"), str(si))
        raw = model_from(
            nz(bank_path(setting, j if is_subset else None) / f"raw_{step}.npz"), str(si)
        )
        independent = sparse_parent(raw) if is_subset else raw
        verify_normalizers(raw, data["color"][mask], data["target"][mask])
        path, pp = (
            RUN / "diagnosed_models" / role / f"{rec['name']}.npz",
            RUN / "diagnosed_predictions" / role / f"{rec['name']}.npz",
        )
        assert sha(path) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
        model, saved = nz(path), nz(pp)
        cap = inspect_export(independent, model, j)
        close({k: rec[k] for k in cap}, cap)
        np.testing.assert_array_equal(saved["row_indices"], query)
        cache_key = (role, family, step, si)
        if cache_key not in cache:
            xx = [transformed(data["color"][query], amount, anchor) for amount, anchor in SETTINGS]
            cache[cache_key] = (xx, np.stack([direct(parent, x) for x in xx]))
        xx, full_output = cache[cache_key]
        expected = (
            np.stack([direct(independent, x)[:, j - 1] for x in xx])
            if is_subset
            else full_output[:, :, j - 1]
        )
        maximum = max(maximum, float(np.max(abs(saved["predictions"] - expected))))
        np.testing.assert_allclose(saved["predictions"], expected, rtol=0, atol=2e-8)
        if is_subset:
            consumer = Predictor(model)
            for x, target in zip(xx, expected, strict=True):
                actual = np.stack([consumer(row) for row in x])
                maximum = max(maximum, float(np.max(abs(actual - target))))
                np.testing.assert_allclose(actual, target, rtol=0, atol=2e-8)
                counts["consumer_calls"] += len(query)
        close(rec["metrics"], full_metrics(expected[0], data, query))
        summaries(
            rec,
            expected,
            data["target"][query],
            data["patient"][query],
            data["device"][query],
            None,
        )
        target_theta = parent["theta"][raw["block_indices"]] if is_subset else parent["theta"]
        raw_guard = guards({"theta": raw["theta"]}, {"theta": target_theta})
        close(rec["raw_comparison"], raw_guard)
        # Build the expected exported arrays explicitly from independent original weights.
        matched = {k: v.copy() for k, v in model.items()}
        h, d = SPECS[family][2], SPECS[family][1]
        ids = [j - 1] if family == "blind4" else list(range(j))
        for b, original in enumerate(ids):
            theta = parent["theta"][original]
            matched[f"w{b}"] = theta[: d * h].reshape(d, h)[: 36 if b == 0 else d].copy()
            matched[f"b{b}"] = theta[d * h : (d + 1) * h].copy()
            matched[f"v{b}"] = theta[(d + 1) * h : -3].reshape(h, 3).copy()
            matched[f"c{b}"] = theta[-3:].copy()
        comparison = compared(model, matched, expected, full_output[:, :, j - 1], data, query)
        comparison["all_guards_pass"] &= raw_guard["weight_guard"]
        close(rec["full_control_comparison"], comparison)
        assert rec["selected_step"] == (step == setting["step"])
        if rec["np_comparison"] is not None:
            np_rec = rec["np_comparison"]
            oldpath = NP / "selected" / role / f"{np_rec['parent_name']}.npz"
            assert sha(oldpath) == np_rec["parent_model_sha256"]
            old = nz(oldpath)
            oldout = nz(NP / "evaluated" / role / f"{np_rec['parent_name']}.npz")["predictions"]
            checked = compared(model, old, expected, oldout, data, query)
            close({k: np_rec[k] for k in checked}, checked)
            counts["np_checks"] += 1
            counts["np_guard_pass"] += int(checked["all_guards_pass"])
        if is_subset:
            counts["subset_models"] += 1
            counts["subset_bitwise"] += int(raw_guard["bitwise_equal"])
            counts["subset_guard_pass"] += int(comparison["all_guards_pass"])
            if rec["selected_step"]:
                counts["selected_subset"] += 1
                counts["selected_subset_guard_pass"] += int(comparison["all_guards_pass"])
        counts["models"] += 1
        counts["prediction_rows"] += len(query) * 33
        counts["transforms"] += 33
        counts["doses"] += 4
        for p in (path, pp):
            artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
        if counts["models"] % 27 == 0:
            print("NB AUDIT", counts["models"], "/243", flush=True)
    assert counts["banks"] == 27 and counts["models"] == 243 and counts["subset_models"] == 162
    assert counts["selected_subset"] == 54 and counts["np_checks"] == 81
    assert counts["prediction_rows"] == 3202254 and counts["consumer_calls"] == 2134836
    assert counts["transforms"] == 8019 and counts["doses"] == 972
    assert counts["subset_guard_pass"] < 162
    for name in ("source_lock.json", "diagnostic_source_lock.json", "diagnostic_results.json"):
        path = RUN / name
        artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
    dependencies = {
        f"scripts/{name}.py": sha(ROOT / f"scripts/{name}.py")
        for name in (
            "chromaseed_neural_blocks_audit",
            "chromaseed_local_denoise_audit",
            "chromaseed_neural_prefix_audit",
            "chromaseed_affine_audit",
            "chromaseed_gate_stability_audit",
        )
    }
    value = dict(
        passed=True,
        equivalence_accepted=False,
        primary_exit_code=1,
        source_lock_sha256=sha(RUN / "source_lock.json"),
        diagnostic_results_sha256=sha(RUN / "diagnostic_results.json"),
        counts=counts,
        maximum_native_lab_audit_drift=maximum,
        artifact_sha256=artifacts,
        dependencies=dependencies,
        seconds=time.perf_counter() - start,
    )
    write_json(OUT / "audit.json", value)
    print("NB AUDIT VALID; EQUIVALENCE REJECTED", counts, flush=True)


if __name__ == "__main__":
    main()
