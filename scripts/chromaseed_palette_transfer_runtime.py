"""P3 actual CPU responses and complete original/aligned/shuffled bank replays."""

from __future__ import annotations

import platform
import time

import numpy as np
import torch
from chromaseed_gated_audit import model_from
from chromaseed_head_range import predict_torch
from chromaseed_head_range_audit import bank_path as prior_bank
from chromaseed_head_range_runtime import response
from chromaseed_kernel_audit import nz
from chromaseed_local_denoise_fit import fit as upstream_fit
from chromaseed_long_training_run import ND, NP, context
from chromaseed_neural_prefix_numpy import export_prefix
from chromaseed_palette_transfer_audit import bank_path
from chromaseed_palette_transfer_fit import fit
from chromaseed_palette_transfer_verification import (
    ARMS,
    CONTRACT,
    HR_OUT,
    OUT,
    PARAMETERS,
    RATES,
    ROLES,
    ROOT,
    RUN,
    RUNTIME_COUNTS,
    SEEDS,
    check_hashes,
    digest,
    encoders_for,
    index_records,
    payload,
    primary_gate,
    read,
    remember,
    require_quiet_host,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from chromaseed_widen_run import load_data
from skin_local_search_train import weights_for
from threadpoolctl import threadpool_info, threadpool_limits


def binding():
    return dict(
        verification_protocol_sha256=digest(CONTRACT),
        results_sha256=digest(RUN / "results.json"),
        audit_sha256=digest(OUT / "audit.json"),
    )


def existing(path, contract):
    if not path.exists():
        return None
    row = read(path)
    assert (
        row["passed"] and row["binding"] == binding() and row["dependencies"] == contract["sources"]
    )
    return row["value"]


def save_measurement(path, value, contract):
    write_once(
        path, dict(passed=True, binding=binding(), dependencies=contract["sources"], value=value)
    )


def replay(data, role, variant, mode, arm, choice, records):
    ix, query, warm_expected, _ = context(data, role)
    prior = read(NP / "selections.json")["roles"][role]["blind4"]
    prefix = prior["policies"]["quality"]["prefix"]
    original = [nz(ND / "selected" / role / f"blind4_s{s}.npz") for s in SEEDS]
    weights = weights_for(data["patient"][ix], data["site"][ix])
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
    upstream_seconds = time.perf_counter() - began
    encoders = encoders_for(arm, {})
    fitted, info = fit(
        data["color"][ix],
        data["tokens"][ix],
        data["target"][ix],
        weights,
        warm,
        variant,
        mode,
        arm,
        encoders,
        choice["step"],
        (choice["step"],),
        "cuda",
        "cuda_graph",
    )
    build_seconds = time.perf_counter() - began
    path = prior_bank(role, variant, mode) if arm == "original" else bank_path(role, variant, arm)
    expected_bank = nz(path / f"models_{choice['step']}.npz")
    assert len(fitted[choice["step"]]) == 6
    for slot, model in enumerate(fitted[choice["step"]]):
        exact(model, model_from(expected_bank, str(slot)))
    checks, pair = [], variant + "__" + arm
    for si, seed in enumerate(SEEDS):
        rec = payload(records[role, pair, seed])
        model = fitted[choice["step"]][2 * si + RATES.index(choice["lr"])]
        exact(model, nz(ROOT / rec["model"]))
        passes = predict_torch(model, data["color"][query], data["tokens"][query], all_passes=True)
        expected = nz(ROOT / rec["output"])
        np.testing.assert_array_equal(expected["row_indices"], query)
        np.testing.assert_array_equal(passes, expected["passes"])
        np.testing.assert_array_equal(passes[:, -1], expected["predictions"])
        checks.append(dict(seed=seed, model_and_prediction_bitwise=True, all_passes_bitwise=True))
    return dict(
        role=role,
        variant=pair,
        architecture=variant,
        head_mode=mode,
        initialization=arm,
        step=choice["step"],
        lr=choice["lr"],
        upstream_seconds=upstream_seconds,
        upstream=upstream,
        continuation=info,
        build_seconds=build_seconds,
        validation_inclusive_seconds=time.perf_counter() - began,
        checks=checks,
        full_bank_slots=6,
        full_replays=1,
        timing_scope="Complete three-parent upstream + six continuation slots, prepared native descriptors; P2 auxiliary training separate, never divide by six",
    )


def main():
    primary_gate()
    assert not (OUT / "verification.json").exists(), "Use report to recheck sealed P3"
    contract = verify_contract()
    audit = verify_stage("audit.json")
    if (OUT / "runtime.json").exists():
        verify_stage("runtime.json")
        print("P3 EXISTING RUNTIME VERIFIED", flush=True)
        return
    require_quiet_host()
    setup("cuda")
    began = time.perf_counter()
    data = load_data()
    results, selection = read(RUN / "results.json"), read(RUN / "selections.json")
    records = index_records(results["records"])
    responses, recipes, artifacts = [], [], {}
    with threadpool_limits(limits=1):
        host = dict(
            platform=platform.platform(),
            torch=torch.__version__,
            numpy=np.__version__,
            gpu=torch.cuda.get_device_name(),
            torch_threads=torch.get_num_threads(),
            threadpools=threadpool_info(),
        )
        training_host = read(RUN / "source_lock.json")
        assert all(host[k] == training_host[k] for k in ("torch", "numpy", "gpu"))
        assert host["torch_threads"] == 1 and all(
            p["num_threads"] == 1 for p in host["threadpools"]
        )
        for rec in results["records"]:
            if rec["architecture"] not in PARAMETERS:
                continue
            path = (
                RUN
                / "verification_v1/runtime/responses"
                / rec["role"]
                / f"{rec['variant']}_s{rec['seed']}.json"
            )
            value = existing(path, contract)
            source = payload(rec)
            if value is None:
                require_quiet_host()
                model, saved = nz(ROOT / source["model"]), nz(ROOT / source["output"])
                ix = saved["row_indices"]
                timing = response(
                    model, data["color"][ix], data["tokens"][ix], saved["predictions"]
                )
                value = dict(
                    role=rec["role"],
                    variant=rec["variant"],
                    seed=rec["seed"],
                    model_sha256=source["model_sha256"],
                    initialization=rec["initialization"],
                    weight_provenance="sealed AS/HR control"
                    if rec["initialization"] == "original"
                    else "new P3 fit",
                    timing_provenance="new P3 measurement",
                    **timing,
                )
                save_measurement(path, value, contract)
            assert (value["role"], value["variant"], value["seed"]) == (
                rec["role"],
                rec["variant"],
                rec["seed"],
            )
            assert value["model_sha256"] == source["model_sha256"]
            assert value["calls"] == 192 and value["distinct_rows"] == 64 and value["warmups"] == 20
            remember(path, artifacts)
            responses.append(value)
        assert len(index_records(responses)) == 81
        print("P3 RESPONSE 81 complete", flush=True)
        for role in ROLES:
            for variant in PARAMETERS:
                mode = training_host["heads"][role][variant]
                for arm in ARMS:
                    pair = variant + "__" + arm
                    choice = selection["roles"][role]["policies"]["per_pair"][pair]
                    path = RUN / "verification_v1/runtime/recipes" / role / f"{pair}.json"
                    value = existing(path, contract)
                    if value is None:
                        require_quiet_host()
                        value = replay(data, role, variant, mode, arm, choice, records)
                        save_measurement(path, value, contract)
                    assert (value["role"], value["variant"], value["step"], value["lr"]) == (
                        role,
                        pair,
                        choice["step"],
                        choice["lr"],
                    )
                    assert value["head_mode"] == mode and value["initialization"] == arm
                    assert value["full_bank_slots"] == 6 and value["full_replays"] == 1
                    assert len(value["upstream"]) == len(value["checks"]) == 3
                    assert all(
                        r["model_and_prediction_bitwise"] and r["all_passes_bitwise"]
                        for r in value["checks"]
                    )
                    remember(path, artifacts)
                    recipes.append(value)
                    print(
                        "P3 RECIPE",
                        role,
                        pair,
                        choice["step"],
                        round(value["build_seconds"], 3),
                        flush=True,
                    )
    counts = dict(
        responses=len(responses),
        timed_calls=sum(r["calls"] for r in responses),
        upstream_fits=sum(len(r["upstream"]) for r in recipes),
        continuation_banks=len(recipes),
        selected_payloads_bitwise=sum(len(r["checks"]) for r in recipes),
        all_bank_payloads_bitwise=sum(r["full_bank_slots"] for r in recipes),
    )
    assert counts == RUNTIME_COUNTS
    primary_gate()
    check_hashes(
        {
            **read(RUN / "source_lock.json")["bindings"],
            **contract["sources"],
            **artifacts,
            **audit["artifact_sha256"],
        }
    )
    write_once(
        OUT / "runtime.json",
        dict(
            passed=True,
            **binding(),
            responses=responses,
            recipes=recipes,
            counts=counts,
            host=host,
            artifact_sha256=artifacts,
            dependencies=contract["sources"],
            invocation_seconds=time.perf_counter() - began,
            inherited_control_timing_source=str(HR_OUT / "runtime.json"),
            inherited_control_timing_sha256=digest(HR_OUT / "runtime.json"),
            scope="81 new CPU responses and 27 complete native-bank replays; NP/WE not remeasured; P2 auxiliary fitting cost separate",
        ),
    )
    print("P3 RUNTIME PASSED", counts, flush=True)


if __name__ == "__main__":
    main()
