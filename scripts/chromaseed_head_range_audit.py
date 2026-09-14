"""Independent HR row, sampler, export, all-pass and frozen-selection audit."""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import torch
from chromaseed_architecture_scale import base_model
from chromaseed_gated_audit import model_from
from chromaseed_head_range import Predictor, predict
from chromaseed_head_range_verification import (
    AS_OUT,
    AS_RUN,
    CONTRACT,
    FILES,
    MODES,
    OUT,
    PARAMETERS,
    RATES,
    ROLES,
    ROOT,
    RUN,
    SEEDS,
    SOURCE,
    TIMES,
    check_hashes,
    compare,
    digest,
    index_records,
    read,
    remember,
    require_terminal,
    select_policies,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_kernel_audit import nz
from chromaseed_local_denoise_audit import close, verify_normalizers
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from chromaseed_widen_run import load_data
from skin_local_search_train import folds_for, roles
from threadpoolctl import threadpool_limits


def bank_path(role, variant, mode, fold=None):
    assert role in ROLES and variant in PARAMETERS and mode in MODES
    root = AS_RUN if mode == "unit" else RUN
    pair = variant if mode == "unit" else variant + "__" + mode
    return (
        root
        / ("final" if fold is None else "inner")
        / role
        / pair
        / ("bank" if fold is None else f"fold{fold}")
    )


def inspect(data, role, variant, mode, fold, artifacts, steps=2048):
    assert mode in ("wide", "linear") and (fold is None or fold in range(3))
    mask, held = roles(data["patient"], data["device"])[role]
    ix, query = np.flatnonzero(mask), np.flatnonzero(held)
    if fold is not None:
        assignment = folds_for(data["patient"][ix], data["device"][ix])
        query, ix = ix[assignment == fold], ix[assignment != fold]
    assert len(ix) and len(query)
    assert not set(data["patient"][ix]) & set(data["patient"][query])
    path = bank_path(role, variant, mode, fold)
    receipt = read(path / "receipt.json")
    remember(path / "receipt.json", artifacts)
    assert receipt["source_lock_sha256"] == SOURCE
    assert receipt["selection_sha256"] == (
        None if fold is not None else digest(RUN / "selections.json")
    )
    assert (receipt["role"], receipt["variant"], receipt["head_mode"], receipt["fold"]) == (
        role,
        variant,
        mode,
        fold,
    )
    assert receipt["steps"] == steps and receipt["horizon"] == 8192 and receipt["batch_size"] == 64
    assert receipt["slots"] == [[s, r] for s in SEEDS for r in RATES]
    checkpoints = list(TIMES if fold is not None else (steps,))
    assert receipt["checkpoints"] == checkpoints
    assert receipt["deployed_parameters"] == PARAMETERS[variant]
    assert receipt["trainable_parameters"] == PARAMETERS[variant] - 643
    assert receipt["engine"] == "cuda_graph" and receipt["device"] == "cuda"
    assert 0 < receipt["cuda_peak_allocated_bytes"] <= 6.5e9
    assert 0 <= receipt["setup_seconds"] <= receipt["full_bank_seconds"]
    assert receipt["full_bank_seconds"] <= receipt["write_and_prediction_inclusive_seconds"]
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
    for name, value in receipt["files"].items():
        remember(path / name, artifacts, value)
    rows = nz(path / "rows.npz")
    assert set(rows) == {"fit_rows", "query_rows"}
    np.testing.assert_array_equal(rows["fit_rows"], ix)
    np.testing.assert_array_equal(
        rows["query_rows"], query if fold is not None else np.array([], np.int64)
    )
    # Parent rows and payloads were separately sealed in AS. No runner selection helper is used.
    prior = bank_path(role, variant, "unit", fold)
    old = read(prior / "receipt.json")
    assert old["warm_parent_sha256"] == receipt["warm_parent_sha256"]
    check_hashes(receipt["warm_parent_sha256"])
    for name in ("receipt.json", "warm_models.npz", "rows.npz"):
        remember(prior / name, artifacts)
    exact(rows, nz(prior / "rows.npz"))
    warm = nz(path / "warm_models.npz")
    exact(warm, nz(prior / "warm_models.npz"))
    parents = [model_from(warm, str(i)) for i in range(3)]
    for parent in parents:
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
    return ix, query, parents, tm, ts


def check_model(model, variant, mode, parent, tm, ts):
    assert str(model["variant"]) == variant and str(model["head_mode"]) == mode
    assert set(model) == {"variant", "head_mode", "theta", "t_mean", "t_std"} | {
        "base_" + k for k in parent
    }
    assert model["theta"].dtype == np.float32 and model["theta"].shape == (
        PARAMETERS[variant] - 643,
    )
    assert np.isfinite(model["theta"]).all()
    exact(base_model(model), parent)
    np.testing.assert_array_equal(model["t_mean"], tm)
    np.testing.assert_array_equal(model["t_std"], ts)
    assert model["t_mean"].dtype == model["t_std"].dtype == np.float32
    assert (
        sum(v.nbytes for v in model.values() if v.dtype.kind in "biufc")
        == 4 * PARAMETERS[variant] + 458
    )


def check_bank(packed, variant, mode, warm, tm, ts):
    assert {k.split("__", 1)[0] for k in packed} == {str(i) for i in range(6)}
    for slot in range(6):
        check_model(model_from(packed, str(slot)), variant, mode, warm[slot // 2], tm, ts)


def audit_inner(data, role, variant, mode, fold, artifacts):
    _, query, warm, tm, ts = inspect(data, role, variant, mode, fold, artifacts)
    path = bank_path(role, variant, mode, fold)
    maximum, predictions = 0.0, {}
    for step in TIMES:
        packed, saved = nz(path / f"models_{step}.npz"), nz(path / f"oof_{step}.npz")
        check_bank(packed, variant, mode, warm, tm, ts)
        assert set(saved) == {"row_indices", "predictions"}
        np.testing.assert_array_equal(saved["row_indices"], query)
        assert saved["predictions"].shape == (6, len(query), 3)
        for slot in range(6):
            model = model_from(packed, str(slot))
            out = predict(model, data["color"][query], data["tokens"][query])
            maximum = max(maximum, compare(out, saved["predictions"][slot]))
        predictions[step] = saved["predictions"]
    return query, predictions, maximum


def candidates_for(data, role, variant, mode, artifacts, counts):
    parts, rows, maximum = {step: [] for step in TIMES}, [], 0.0
    for fold in range(3):
        query, predictions, drift = audit_inner(data, role, variant, mode, fold, artifacts)
        maximum = max(maximum, drift)
        rows.append(query)
        for step in TIMES:
            parts[step].append(predictions[step])
        counts["inner_banks"] += 1
        counts["inner_models"] += 18
        counts["inner_vectors"] += 18 * len(query)
        print("HR AUDIT INNER", role, variant, mode, fold, "maxLab", drift, flush=True)
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
            mm = [
                error_summary(p, data["target"][full], data["patient"][full], data["site"][full])[0]
                for p in predictions
            ]
            candidates.append(
                dict(
                    variant=variant + "__" + mode,
                    architecture=variant,
                    head_mode=mode,
                    step=step,
                    lr=rate,
                    parameters=PARAMETERS[variant],
                    numeric_bytes=4 * PARAMETERS[variant] + 458,
                    clean=float(np.mean([m["person_mean"] for m in mm])),
                    p90=float(np.mean([m["p90"] for m in mm])),
                    seed_metrics=mm,
                )
            )
    return candidates, maximum


def audit_final(data, role, variant, mode, policies, records, artifacts, counts):
    pair = variant + "__" + mode
    choice = policies["per_pair"][pair]
    _, query, warm, tm, ts = inspect(data, role, variant, mode, None, artifacts, choice["step"])
    path = bank_path(role, variant, mode) / f"models_{choice['step']}.npz"
    packed = nz(path)
    check_bank(packed, variant, mode, warm, tm, ts)
    counts["final_bank_payloads"] += 6
    maximum = 0.0
    for si, seed in enumerate(SEEDS):
        rec = records[role, pair, seed]
        slot = 2 * si + RATES.index(choice["lr"])
        assert rec["architecture"] == variant and rec["head_mode"] == mode
        assert rec["slot"] == slot and rec["source_path"] == path.relative_to(ROOT).as_posix()
        assert rec["source_sha256"] == digest(path)
        assert rec["step"] == choice["step"] and rec["lr"] == choice["lr"]
        assert rec["overall"] == (policies["overall"]["variant"] == pair)
        for key in ("model", "output"):
            remember(ROOT / rec[key], artifacts, rec[key + "_sha256"])
        model, saved = nz(ROOT / rec["model"]), nz(ROOT / rec["output"])
        exact(model, model_from(packed, str(slot)))
        assert set(saved) == {"row_indices", "predictions", "passes"}
        np.testing.assert_array_equal(saved["row_indices"], query)
        n_passes = 4 if variant.startswith(("soft", "dynamic")) else 1
        assert saved["passes"].shape == (len(query), n_passes, 3)
        np.testing.assert_array_equal(saved["predictions"], saved["passes"][:, -1])
        consumer = Predictor(model)
        for i, row in enumerate(query):
            out = consumer(data["color"][row], data["tokens"][row], all_passes=True)
            maximum = max(maximum, compare(out, saved["passes"][i]))
            counts["actual_single_calls"] += 1
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
    print("HR AUDIT FINAL", role, pair, "maxLab", maximum, flush=True)
    return maximum


def probe(paths):
    """Fixed completed-inner snapshot; cannot create a main audit or quality result."""
    assert not torch.cuda.is_initialized()
    setup("cpu")
    paths = sorted({(RUN / p).resolve() for p in paths})
    for path in paths:
        if not path.is_relative_to((RUN / "inner").resolve()):
            raise ValueError("Probe paths must be completed inner-bank directories")
    assert paths
    sources = {name: digest(ROOT / name) for name in FILES}
    contract = dict(
        scope="partial completed-inner probe; no selection, final audit or quality conclusion",
        sources=sources,
        source_lock_sha256=digest(RUN / "source_lock.json"),
        receipts={str(p / "receipt.json"): digest(p / "receipt.json") for p in paths},
    )
    assert contract["source_lock_sha256"] == SOURCE
    key = hashlib.sha256(str(contract).encode()).hexdigest()[:16]
    destination = RUN / "verification_v1/probes" / key
    write_once(destination / "protocol.json", contract)
    if (destination / "result.json").exists():
        result = read(destination / "result.json")
        check_hashes({**sources, **result["artifact_sha256"]})
        assert result["protocol_sha256"] == digest(destination / "protocol.json")
        print("HR EXISTING PARTIAL PROBE VERIFIED", digest(destination / "result.json"), flush=True)
        return
    started = time.perf_counter()
    artifacts, records = {}, []
    data = load_data()
    with threadpool_limits(limits=1):
        for path in paths:
            receipt = read(path / "receipt.json")
            query, _, maximum = audit_inner(
                data,
                receipt["role"],
                receipt["variant"],
                receipt["head_mode"],
                receipt["fold"],
                artifacts,
            )
            records.append(
                dict(path=str(path), models=18, vectors=18 * len(query), maximum_native_lab=maximum)
            )
            print("HR PARTIAL PROBE", str(path), maximum, flush=True)
    check_hashes({**sources, **artifacts, **contract["receipts"]})
    assert not torch.cuda.is_initialized()
    result = dict(
        passed=True,
        full_primary_audit=False,
        quality_improvement_claim=False,
        protocol_sha256=digest(destination / "protocol.json"),
        records=records,
        models=sum(r["models"] for r in records),
        vectors=sum(r["vectors"] for r in records),
        maximum_native_lab=max(r["maximum_native_lab"] for r in records),
        artifact_sha256=artifacts,
        seconds=time.perf_counter() - started,
    )
    write_once(destination / "result.json", result)
    print(
        "HR PARTIAL PROBE PASSED",
        digest(destination / "result.json"),
        result["models"],
        result["vectors"],
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-bank",
        action="append",
        help="Completed path relative to HR RUN; repeat to fix a small snapshot",
    )
    args = parser.parse_args()
    if args.probe_bank:
        probe(args.probe_bank)
        return
    # Deliberately before source/input loading or any write.
    require_terminal(read(RUN / "job.json"))
    assert not (OUT / "verification.json").exists(), (
        "Sealed HR: use report for read-only verification"
    )
    contract = verify_contract()
    if (OUT / "audit.json").exists():
        verify_stage("audit.json")
        print("HR EXISTING FULL AUDIT VERIFIED", flush=True)
        return
    setup("cpu")
    assert not torch.cuda.is_initialized()
    started = time.perf_counter()
    selection, results = read(RUN / "selections.json"), read(RUN / "results.json")
    assert selection["source_lock_sha256"] == results["source_lock_sha256"] == SOURCE
    assert results["selection_sha256"] == digest(RUN / "selections.json")
    assert read(RUN / "job.json")["results_sha256"] == digest(RUN / "results.json")
    previous, oldresults = read(AS_RUN / "selections.json"), read(AS_RUN / "results.json")
    oldrecords, records = index_records(oldresults["records"]), index_records(results["records"])
    variants = tuple(v + "__" + m for v in PARAMETERS for m in MODES) + ("np", "we")
    assert set(records) == {(role, v, s) for role in ROLES for v in variants for s in SEEDS}
    assert sum(r["overall"] for r in records.values()) == 9
    assert set(selection["roles"]) == set(ROLES)
    artifacts = dict(contract["recovery_bindings"])
    for name in ("source_lock.json", "selections.json", "results.json", "job.json"):
        remember(RUN / name, artifacts)
    remember(CONTRACT, artifacts)
    remember(AS_OUT / "verification.json", artifacts)
    audit_protocol = RUN / "verification_v1/full_audit_protocol.json"
    write_once(
        audit_protocol,
        dict(
            verification_protocol_sha256=digest(CONTRACT),
            input_sha256={
                str(RUN / name): digest(RUN / name)
                for name in ("source_lock.json", "selections.json", "results.json", "job.json")
            },
            expected_counts=contract["expected_counts"],
            scope="full verification of these terminal primary artifacts; no fitting or reselection",
        ),
    )
    remember(audit_protocol, artifacts)
    counts = dict.fromkeys(contract["expected_counts"], 0)
    maximum = 0.0
    data = load_data()
    assert len(data["target"]) == 966 and len(np.unique(data["patient"])) == 24
    with threadpool_limits(limits=1):
        for role in ROLES:
            candidates = []
            for variant in PARAMETERS:
                for mode in MODES:
                    if mode == "unit":
                        old = [
                            c
                            for c in previous["roles"][role]["candidates"]
                            if c["variant"] == variant
                        ]
                        assert len(old) == 6
                        candidates.extend(
                            dict(
                                c,
                                variant=variant + "__unit",
                                architecture=variant,
                                head_mode="unit",
                            )
                            for c in old
                        )
                        counts["inherited_candidates"] += len(old)
                    else:
                        new, drift = candidates_for(data, role, variant, mode, artifacts, counts)
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
                for mode in ("wide", "linear"):
                    maximum = max(
                        maximum,
                        audit_final(
                            data, role, variant, mode, policies, records, artifacts, counts
                        ),
                    )
            for variant in (*PARAMETERS, "np", "we"):
                pair = variant + "__unit" if variant in PARAMETERS else variant
                for seed in SEEDS:
                    old = oldrecords[role, variant, seed]
                    expected = dict(
                        role=role,
                        variant=pair,
                        architecture=variant,
                        head_mode="unit" if variant in PARAMETERS else "control",
                        seed=seed,
                        step=old["step"],
                        lr=old["lr"],
                        inherited_as_record=old,
                        metrics=old["metrics"],
                        overall=policies["overall"]["variant"] == pair,
                    )
                    assert records[role, pair, seed] == expected
                    counts["inherited_records"] += 1
    assert counts == contract["expected_counts"], counts
    assert not torch.cuda.is_initialized()
    require_terminal(read(RUN / "job.json"))
    verify_contract(full_inputs=False)
    check_hashes(artifacts)
    value = dict(
        passed=True,
        verification_protocol_sha256=digest(CONTRACT),
        results_sha256=digest(RUN / "results.json"),
        counts=counts,
        maximum_native_lab=maximum,
        artifact_sha256=artifacts,
        dependencies=contract["sources"],
        seconds=time.perf_counter() - started,
        scope="independent NumPy and metrics for new HR fits; exact remapping of sealed AS controls",
    )
    write_once(OUT / "audit.json", value)
    print("HR FULL AUDIT PASSED", counts, "maxLab", maximum, flush=True)


if __name__ == "__main__":
    main()
