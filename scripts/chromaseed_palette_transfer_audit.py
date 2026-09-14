"""Independent full P3 initialization, rows, exports, selections and all-pass audit."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from chromaseed_gated_audit import model_from
from chromaseed_head_range import Predictor, predict
from chromaseed_head_range_audit import bank_path as prior_bank
from chromaseed_head_range_audit import check_model as check_hr_model
from chromaseed_kernel_audit import nz
from chromaseed_local_denoise_audit import close, verify_normalizers
from chromaseed_palette_transfer_verification import (
    ARMS,
    CONTRACT,
    ENCODER_PARAMETERS,
    EXPECTED,
    FILES,
    HR_OUT,
    HR_RUN,
    NATIVE_INPUTS,
    OUT,
    PARAMETERS,
    RATES,
    REGISTRATION,
    ROLES,
    ROOT,
    RUN,
    SEEDS,
    TIMES,
    check_hashes,
    compare,
    digest,
    encoder_content,
    encoders_for,
    index_records,
    initial_theta,
    primary_gate,
    read,
    registration_check,
    remember,
    select_policies,
    selected_heads,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from chromaseed_widen_run import load_data
from skin_local_search_train import folds_for, roles
from threadpoolctl import threadpool_limits


def bank_path(role, variant, arm, fold=None):
    assert role in ROLES and variant in PARAMETERS and arm in ARMS[1:]
    return (
        RUN
        / ("final" if fold is None else "inner")
        / role
        / (variant + "__" + arm)
        / ("bank" if fold is None else f"fold{fold}")
    )


def inspect(data, role, variant, mode, arm, fold, artifacts, counts, steps=2048):
    mask, held = roles(data["patient"], data["device"])[role]
    ix, query = np.flatnonzero(mask), np.flatnonzero(held)
    if fold is not None:
        assignment = folds_for(data["patient"][ix], data["device"][ix])
        query, ix = ix[assignment == fold], ix[assignment != fold]
    assert len(ix) and len(query) and not set(data["patient"][ix]) & set(data["patient"][query])
    path = bank_path(role, variant, arm, fold)
    receipt = read(path / "receipt.json")
    remember(path / "receipt.json", artifacts)
    assert receipt["source_lock_sha256"] == digest(RUN / "source_lock.json")
    assert receipt["selection_sha256"] == (
        None if fold is not None else digest(RUN / "selections.json")
    )
    assert (
        receipt["role"],
        receipt["variant"],
        receipt["head_mode"],
        receipt["initialization"],
        receipt["fold"],
    ) == (role, variant, mode, arm, fold)
    assert receipt["steps"] == steps and receipt["horizon"] == 8192 and receipt["batch_size"] == 64
    assert receipt["slots"] == [[s, r] for s in SEEDS for r in RATES]
    checkpoints = list(TIMES if fold is not None else (steps,))
    assert receipt["checkpoints"] == checkpoints
    assert receipt["deployed_parameters"] == PARAMETERS[variant]
    assert receipt["trainable_parameters"] == PARAMETERS[variant] - 643
    assert receipt["engine"] == "cuda_graph" and receipt["device"] == "cuda"
    assert 0 < receipt["cuda_peak_allocated_bytes"] <= 6.5e9
    assert (
        0
        <= receipt["setup_seconds"]
        <= receipt["full_bank_seconds"]
        <= receipt["write_and_prediction_inclusive_seconds"]
    )
    expected_trace = sorted(set(range(128, steps + 1, 128)) | set(checkpoints))
    assert [t["step"] for t in receipt["trace"]] == expected_trace
    elapsed = receipt["setup_seconds"]
    for item in receipt["trace"]:
        assert elapsed <= item["seconds"] <= receipt["full_bank_seconds"]
        losses = np.asarray(item["minibatch_loss"])
        assert losses.shape == (6,) and np.isfinite(losses).all()
        elapsed = item["seconds"]
    names = {"warm_models.npz", "rows.npz"} | {f"models_{s}.npz" for s in checkpoints}
    if fold is not None:
        names |= {f"oof_{s}.npz" for s in checkpoints}
    assert set(receipt["files"]) == names
    for name, sha in receipt["files"].items():
        remember(path / name, artifacts, sha)
    rows = nz(path / "rows.npz")
    assert set(rows) == {"fit_rows", "query_rows"}
    np.testing.assert_array_equal(rows["fit_rows"], ix)
    np.testing.assert_array_equal(
        rows["query_rows"], query if fold is not None else np.array([], np.int64)
    )
    prior = prior_bank(role, variant, "unit", fold)
    old = read(prior / "receipt.json")
    assert old["warm_parent_sha256"] == receipt["warm_parent_sha256"]
    check_hashes(receipt["warm_parent_sha256"])
    artifacts.update(receipt["warm_parent_sha256"])
    for name in ("receipt.json", "warm_models.npz", "rows.npz"):
        remember(prior / name, artifacts)
    exact(rows, nz(prior / "rows.npz"))
    packed_warm = nz(path / "warm_models.npz")
    exact(packed_warm, nz(prior / "warm_models.npz"))
    warm = [model_from(packed_warm, str(i)) for i in range(3)]
    for parent in warm:
        verify_normalizers(parent, data["color"][ix], data["target"][ix])
    tokens = data["tokens"][ix].astype(np.float32)
    assert hashlib.sha256(tokens.tobytes()).hexdigest() == receipt["fit_tokens_sha256"]
    tm = tokens.astype(float).mean((0, 1)).astype(np.float32)
    ts = np.maximum(tokens.astype(float).std((0, 1)), 1e-6).astype(np.float32)
    weights = balanced(data["patient"][ix], data["site"][ix])
    weights /= weights.sum()
    draws = np.stack(
        [np.random.default_rng(s + 830003).choice(len(ix), (steps, 64), p=weights) for s in SEEDS]
    )
    assert hashlib.sha256(draws.tobytes()).hexdigest() == receipt["sampling_sha256"]
    factor = np.float32(0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (steps - 1) / 8192)))
    np.testing.assert_array_equal(
        np.asarray(receipt["final_learning_rates"], np.float32),
        np.tile(np.array(RATES, np.float32), 3) * factor,
    )
    encoders = encoders_for(arm, artifacts)
    expected = initial_theta(variant, arm, encoders, tm, ts)
    assert hashlib.sha256(expected.tobytes()).hexdigest() == receipt["initial_theta_sha256"]
    assert receipt["encoder_digests"] == [encoder_content(e) for e in encoders]
    assert receipt["transferred_parameters"] == ENCODER_PARAMETERS
    counts["initialization_banks"] += 1
    return ix, query, warm, tm, ts, encoders


def check_model(model, variant, mode, arm, encoder, parent, tm, ts):
    assert arm in ARMS[1:]
    assert str(model["palette_arm"]) == arm
    assert str(model["palette_encoder_digest"]) == encoder_content(encoder)
    stripped = {
        k: v for k, v in model.items() if k not in ("palette_arm", "palette_encoder_digest")
    }
    if mode == "unit":
        assert "head_mode" not in stripped
        stripped["head_mode"] = np.asarray("unit")
    check_hr_model(stripped, variant, mode, parent, tm, ts)
    assert (
        sum(v.nbytes for v in model.values() if v.dtype.kind in "biufc")
        == 4 * PARAMETERS[variant] + 458
    )


def check_bank(packed, variant, mode, arm, encoders, warm, tm, ts):
    assert {k.split("__", 1)[0] for k in packed} == {str(i) for i in range(6)}
    for slot in range(6):
        check_model(
            model_from(packed, str(slot)),
            variant,
            mode,
            arm,
            encoders[slot // 2],
            warm[slot // 2],
            tm,
            ts,
        )


def verify_passes(model, x, tokens, expected):
    consumer = Predictor(model)
    maximum = 0.0
    assert len(x) == len(tokens) == len(expected) and len(x) > 0
    for i in range(len(x)):
        actual = consumer(x[i], tokens[i], all_passes=True)
        maximum = max(maximum, compare(actual, expected[i]))
    return maximum, len(x)


def candidates_for(data, role, variant, mode, arm, artifacts, counts):
    parts, rows, maximum = {s: [] for s in TIMES}, [], 0.0
    for fold in range(3):
        _, query, warm, tm, ts, encoders = inspect(
            data, role, variant, mode, arm, fold, artifacts, counts
        )
        path = bank_path(role, variant, arm, fold)
        for step in TIMES:
            packed, saved = nz(path / f"models_{step}.npz"), nz(path / f"oof_{step}.npz")
            check_bank(packed, variant, mode, arm, encoders, warm, tm, ts)
            assert set(saved) == {"row_indices", "predictions"}
            np.testing.assert_array_equal(saved["row_indices"], query)
            assert saved["predictions"].shape == (6, len(query), 3)
            for slot in range(6):
                out = predict(
                    model_from(packed, str(slot)), data["color"][query], data["tokens"][query]
                )
                maximum = max(maximum, compare(out, saved["predictions"][slot]))
            parts[step].append(saved["predictions"])
        rows.append(query)
        counts["inner_banks"] += 1
        counts["inner_models"] += 18
        counts["inner_vectors"] += 18 * len(query)
        print("P3 AUDIT INNER", role, variant, arm, fold, "maxLab", maximum, flush=True)
    rows = np.concatenate(rows)
    order = np.argsort(rows)
    full = np.flatnonzero(roles(data["patient"], data["device"])[role][0])
    np.testing.assert_array_equal(rows[order], full)
    candidates = []
    for ri, rate in enumerate(RATES):
        for step in TIMES:
            predictions = np.concatenate(parts[step], axis=1)[:, order][
                [2 * i + ri for i in range(3)]
            ]
            metrics = [
                error_summary(p, data["target"][full], data["patient"][full], data["site"][full])[0]
                for p in predictions
            ]
            candidates.append(
                dict(
                    variant=variant + "__" + arm,
                    architecture=variant,
                    head_mode=mode,
                    initialization=arm,
                    step=step,
                    lr=rate,
                    parameters=PARAMETERS[variant],
                    numeric_bytes=4 * PARAMETERS[variant] + 458,
                    clean=float(np.mean([m["person_mean"] for m in metrics])),
                    p90=float(np.mean([m["p90"] for m in metrics])),
                    seed_metrics=metrics,
                )
            )
    return candidates, maximum


def audit_final(data, role, variant, mode, arm, policies, records, artifacts, counts):
    pair = variant + "__" + arm
    choice = policies["per_pair"][pair]
    _, query, warm, tm, ts, encoders = inspect(
        data, role, variant, mode, arm, None, artifacts, counts, choice["step"]
    )
    path = bank_path(role, variant, arm) / f"models_{choice['step']}.npz"
    packed = nz(path)
    check_bank(packed, variant, mode, arm, encoders, warm, tm, ts)
    counts["final_bank_payloads"] += 6
    maximum = 0.0
    for si, seed in enumerate(SEEDS):
        rec = records[role, pair, seed]
        slot = 2 * si + RATES.index(choice["lr"])
        assert (
            rec["architecture"] == variant
            and rec["head_mode"] == mode
            and rec["initialization"] == arm
        )
        assert rec["slot"] == slot and rec["source_path"] == str(path)
        assert rec["source_sha256"] == digest(path)
        assert rec["step"] == choice["step"] and rec["lr"] == choice["lr"]
        assert rec["overall"] == (policies["overall"]["variant"] == pair)
        for key in ("model", "output"):
            remember(Path(rec[key]), artifacts, rec[key + "_sha256"])
        model, saved = nz(rec["model"]), nz(rec["output"])
        exact(model, model_from(packed, str(slot)))
        assert set(saved) == {"row_indices", "predictions", "passes"}
        np.testing.assert_array_equal(saved["row_indices"], query)
        assert saved["passes"].shape == (len(query), 1 if variant == "patch5m" else 4, 3)
        np.testing.assert_array_equal(saved["predictions"], saved["passes"][:, -1])
        drift, calls = verify_passes(
            model, data["color"][query], data["tokens"][query], saved["passes"]
        )
        maximum = max(maximum, drift)
        counts["actual_single_calls"] += calls
        metrics = error_summary(
            saved["predictions"], data["target"][query], data["patient"][query], data["site"][query]
        )[0]
        close(rec["metrics"], metrics)
        assert (
            rec["parameters"] == PARAMETERS[variant]
            and rec["numeric_bytes"] == 4 * PARAMETERS[variant] + 458
        )
        counts["final_models"] += 1
        counts["final_vectors"] += len(query)
    counts["final_banks"] += 1
    print("P3 AUDIT FINAL", role, pair, "maxLab", maximum, flush=True)
    return maximum


def probe_initializers():
    """Actual encoders/native FIT statistics, no native fitting and no production seal."""
    from chromaseed_palette_transfer import Bank

    registration_check()
    assert not torch.cuda.is_initialized()
    setup("cpu")
    sources = {f: digest(ROOT / f) for f in FILES}
    artifacts = {}
    encoders = {a: encoders_for(a, artifacts) for a in ARMS}
    native_inputs = NATIVE_INPUTS
    protocol = dict(
        registration_sha256=REGISTRATION,
        sources=sources,
        encoders=artifacts,
        native_inputs=native_inputs,
        scope="CPU initial-parameter reconstruction with actual P2 encoders/native inner FIT statistics; no fitting or accuracy",
    )
    key = hashlib.sha256(str(protocol).encode()).hexdigest()[:16]
    destination = RUN / "verification_v1/probes" / key
    write_once(destination / "protocol.json", protocol)
    if (destination / "result.json").exists():
        result = read(destination / "result.json")
        check_hashes({**sources, **artifacts, **native_inputs})
        assert result["passed"] and result["protocol_sha256"] == digest(
            destination / "protocol.json"
        )
        print(
            "P3 EXISTING INITIALIZER PROBE VERIFIED",
            digest(destination / "result.json"),
            flush=True,
        )
        return
    began = time.perf_counter()
    data = load_data()
    records = []
    with threadpool_limits(limits=1):
        for role in ROLES:
            ix = np.flatnonzero(roles(data["patient"], data["device"])[role][0])
            assignments = folds_for(data["patient"][ix], data["device"][ix])
            ix = ix[assignments != 0]
            tokens = data["tokens"][ix].astype(np.float32)
            mean = tokens.astype(float).mean((0, 1)).astype(np.float32)
            std = np.maximum(tokens.astype(float).std((0, 1)), 1e-6).astype(np.float32)
            for variant in PARAMETERS:
                for arm in ARMS:
                    expected = initial_theta(variant, arm, encoders[arm], mean, std)
                    modes = []
                    for mode in ("unit", "wide", "linear"):
                        net = Bank(variant, mode, arm, encoders[arm], mean, std)
                        np.testing.assert_array_equal(expected, net.theta.detach().numpy())
                        modes.append(mode)
                        del net
                    records.append(
                        dict(
                            role=role,
                            variant=variant,
                            arm=arm,
                            modes=modes,
                            slots=6,
                            initial_theta_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                        )
                    )
                    del expected
    check_hashes({**sources, **artifacts, **native_inputs})
    assert not torch.cuda.is_initialized()
    result = dict(
        passed=True,
        protocol_sha256=digest(destination / "protocol.json"),
        records=records,
        configurations=len(records) * 3,
        slots_checked=len(records) * 18,
        seconds=time.perf_counter() - began,
        native_finetuning_performed=False,
        native_accuracy_claim=False,
        full_primary_audit=False,
    )
    write_once(destination / "result.json", result)
    print(
        "P3 INITIALIZER PROBE PASSED",
        digest(destination / "result.json"),
        result["configurations"],
        result["slots_checked"],
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-initializers", action="store_true")
    if parser.parse_args().probe_initializers:
        probe_initializers()
        return
    primary_gate()
    assert not (OUT / "verification.json").exists(), "Use report to recheck sealed P3"
    contract = verify_contract()
    if (OUT / "audit.json").exists():
        verify_stage("audit.json")
        print("P3 EXISTING AUDIT VERIFIED", flush=True)
        return
    setup("cpu")
    assert not torch.cuda.is_initialized()
    began = time.perf_counter()
    selection, results = read(RUN / "selections.json"), read(RUN / "results.json")
    previous, oldresults = read(HR_RUN / "selections.json"), read(HR_RUN / "results.json")
    heads = selected_heads(previous)
    oldrecords, records = index_records(oldresults["records"]), index_records(results["records"])
    variants = tuple(v + "__" + a for v in PARAMETERS for a in ARMS) + ("np", "we")
    assert set(records) == {(r, v, s) for r in ROLES for v in variants for s in SEEDS}
    assert sum(r["overall"] for r in records.values()) == 9
    assert set(selection["roles"]) == set(ROLES)
    artifacts = dict(contract["native_input_bindings"])
    for name in ("source_lock.json", "selections.json", "results.json", "job.json"):
        remember(RUN / name, artifacts)
    remember(CONTRACT, artifacts)
    remember(HR_OUT / "verification.json", artifacts)
    protocol = RUN / "verification_v1/full_audit_protocol.json"
    write_once(
        protocol,
        dict(
            verification_protocol_sha256=digest(CONTRACT),
            input_sha256=artifacts.copy(),
            expected_counts=EXPECTED,
        ),
    )
    remember(protocol, artifacts)
    counts, maximum = dict.fromkeys(EXPECTED, 0), 0.0
    data = load_data()
    assert len(data["target"]) == 966 and len(np.unique(data["patient"])) == 24
    with threadpool_limits(limits=1):
        for role in ROLES:
            candidates = []
            for variant in PARAMETERS:
                mode = heads[role][variant]
                for arm in ARMS:
                    if arm == "original":
                        old = [
                            c
                            for c in previous["roles"][role]["candidates"]
                            if c["variant"] == variant + "__" + mode
                        ]
                        assert len(old) == 6
                        candidates.extend(
                            dict(c, variant=variant + "__original", initialization="original")
                            for c in old
                        )
                        counts["inherited_candidates"] += len(old)
                    else:
                        new, drift = candidates_for(
                            data, role, variant, mode, arm, artifacts, counts
                        )
                        candidates.extend(new)
                        maximum = max(maximum, drift)
            candidates.extend(
                next(c for c in previous["roles"][role]["candidates"] if c["variant"] == k)
                for k in ("np", "we")
            )
            close(selection["roles"][role]["candidates"], candidates)
            policies = select_policies(candidates)
            close(selection["roles"][role]["policies"], policies)
            counts["candidates"] += len(candidates)
            counts["choices"] += len(policies["per_pair"]) + len(policies["per_architecture"]) + 1
            for variant in PARAMETERS:
                mode = heads[role][variant]
                original = policies["per_pair"][variant + "__original"]
                prior = previous["roles"][role]["policies"]["per_architecture"][variant]
                assert all(original[k] == prior[k] for k in ("step", "lr", "clean", "head_mode"))
                for arm in ARMS[1:]:
                    maximum = max(
                        maximum,
                        audit_final(
                            data, role, variant, mode, arm, policies, records, artifacts, counts
                        ),
                    )
            for variant in (*PARAMETERS, "np", "we"):
                pair = variant + "__original" if variant in PARAMETERS else variant
                mode = heads[role][variant] if variant in PARAMETERS else "control"
                inherited_pair = variant + "__" + mode if variant in PARAMETERS else variant
                for seed in SEEDS:
                    old = oldrecords[role, inherited_pair, seed]
                    expected = dict(
                        role=role,
                        variant=pair,
                        architecture=variant,
                        head_mode=mode,
                        initialization="original" if variant in PARAMETERS else "control",
                        seed=seed,
                        step=old["step"],
                        lr=old["lr"],
                        inherited_hr_record=old,
                        metrics=old["metrics"],
                        overall=policies["overall"]["variant"] == pair,
                    )
                    assert records[role, pair, seed] == expected
                    counts["inherited_records"] += 1
    assert counts == EXPECTED, counts
    assert not torch.cuda.is_initialized()
    primary_gate()
    check_hashes({**read(RUN / "source_lock.json")["bindings"], **contract["sources"], **artifacts})
    write_once(
        OUT / "audit.json",
        dict(
            passed=True,
            verification_protocol_sha256=digest(CONTRACT),
            results_sha256=digest(RUN / "results.json"),
            counts=counts,
            maximum_native_lab=maximum,
            artifact_sha256=artifacts,
            dependencies=contract["sources"],
            seconds=time.perf_counter() - began,
            scope="Independent initial weights, native rows, all new exports and final single-example passes; sealed HR controls remapped exactly",
        ),
    )
    print("P3 FULL AUDIT PASSED", counts, "maxLab", maximum, flush=True)


if __name__ == "__main__":
    main()
