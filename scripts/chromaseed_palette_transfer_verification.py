"""Independent P3 intervention reconstruction, contracts and complete cost scopes."""

from __future__ import annotations

import argparse
import hashlib
import math
import zlib
from pathlib import Path

import numpy as np
import torch
from chromaseed_head_range_verification import (
    AS_RUN as AS_RUN,
)
from chromaseed_head_range_verification import (
    OUT as HR_OUT,
)
from chromaseed_head_range_verification import (
    RATES as RATES,
)
from chromaseed_head_range_verification import (
    ROLES as ROLES,
)
from chromaseed_head_range_verification import (
    ROOT as ROOT,
)
from chromaseed_head_range_verification import (
    RUN as HR_RUN,
)
from chromaseed_head_range_verification import (
    SEEDS as SEEDS,
)
from chromaseed_head_range_verification import (
    SOURCE as HR_SOURCE,
)
from chromaseed_head_range_verification import (
    TIMES as TIMES,
)
from chromaseed_head_range_verification import (
    check_hashes as check_hashes,
)
from chromaseed_head_range_verification import (
    compare as compare,
)
from chromaseed_head_range_verification import (
    digest as digest,
)
from chromaseed_head_range_verification import (
    index_records as index_records,
)
from chromaseed_head_range_verification import (
    rank,
    require_terminal,
)
from chromaseed_head_range_verification import (
    read as read,
)
from chromaseed_head_range_verification import (
    remember as remember,
)
from chromaseed_head_range_verification import (
    require_quiet_host as require_quiet_host,
)
from chromaseed_head_range_verification import (
    write_once as write_once,
)
from chromaseed_kernel_audit import nz

RUN = Path("D:/Luma-RnD/chromaseed_palette_transfer_v1")
P2 = Path("D:/Luma-RnD/chromaseed_palette_pretrain_v1")
OUT = ROOT / "docs/benchmarks/chromaseed_palette_transfer_v1"
CONTRACT = RUN / "verification_v1/protocol.json"
REGISTRATION = "90a9a78bc46f5ca1f40befe0cd04e8861374868d41710c099c7573da7a2add17"
P2_FIT = "37236660053b4534b7e5fd6d73df306c604bc4a5457aafc4dd12eb79bbae3328"
NATIVE_CACHE_SHA = "d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0"
NATIVE_INPUTS = {
    str(
        ROOT.parents[1] / "luma-skin-vision-rnd/data/processed/skin_mskcc_pixels_v1/train.npz"
    ): NATIVE_CACHE_SHA
}
PARAMETERS = dict(patch5m=4962566, soft5m=4846822, dynamic5m=4846822)
ARMS = ("original", "aligned", "shuffled")
ENCODER_PARAMETERS = 105856
FILES = (
    "scripts/chromaseed_palette_transfer_verification.py",
    "scripts/chromaseed_palette_transfer_audit.py",
    "scripts/chromaseed_palette_transfer_runtime.py",
    "scripts/chromaseed_palette_transfer_report.py",
    "tests/test_chromaseed_palette_transfer_verification.py",
    "docs/research/chromaseed_palette_transfer_verification_v1_protocol.md",
)
EXPECTED = dict(
    inner_banks=54,
    inner_models=972,
    inner_vectors=183600,
    candidates=168,
    choices=39,
    final_banks=18,
    final_bank_payloads=108,
    final_models=54,
    final_vectors=21564,
    actual_single_calls=21564,
    inherited_candidates=54,
    inherited_records=45,
    initialization_banks=72,
)
RUNTIME_COUNTS = dict(
    responses=81,
    timed_calls=15552,
    upstream_fits=81,
    continuation_banks=27,
    selected_payloads_bitwise=81,
    all_bank_payloads_bitwise=162,
)


def validate_encoder(model, seed, arm):
    assert set(model) == {"theta", "mean", "std", "seed", "arm"}
    assert model["seed"].shape == model["arm"].shape == ()
    assert int(model["seed"]) == seed and str(model["arm"]) == arm
    assert model["theta"].shape == (ENCODER_PARAMETERS,) and model["theta"].dtype == np.float32
    assert model["mean"].shape == model["std"].shape == (18,)
    assert model["mean"].dtype == model["std"].dtype == np.float32
    assert all(np.isfinite(model[k]).all() for k in ("theta", "mean", "std"))
    assert (model["std"] > 0).all()


def encoder_content(model):
    digestor = hashlib.sha256()
    for name in sorted(model):
        value = model[name]
        description = f"{name}|{value.dtype.str}|{value.shape}|{value.nbytes}|".encode()
        digestor.update(len(description).to_bytes(8, "little"))
        digestor.update(description)
        digestor.update(value.tobytes())
    return digestor.hexdigest()


def encoders_for(arm, artifacts):
    assert arm in ARMS
    if arm == "original":
        return None
    remember(P2 / "fit.json", artifacts, P2_FIT)
    rows = read(P2 / "fit.json")["exports"]
    assert len(rows) == 6 and {(r["seed"], r["arm"]) for r in rows} == {
        (seed, a) for seed in SEEDS for a in ARMS[1:]
    }
    result = []
    for seed in SEEDS:
        row = next(r for r in rows if (r["seed"], r["arm"]) == (seed, arm))
        remember(Path(row["path"]), artifacts, row["sha256"])
        model = nz(row["path"])
        validate_encoder(model, seed, arm)
        result.append(model)
    return result


def transferred(encoder, mean, std):
    """Independent affine change of coordinates; do not call the P2 transplant helper."""
    mean, std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
    assert mean.shape == std.shape == (18,) and (std > 0).all()
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    result = encoder["theta"].copy()
    original = encoder["theta"].astype(np.float64)
    weights = original[:6912].reshape(18, 384)
    result[:6912] = ((std.astype(np.float64) / encoder["std"])[:, None] * weights).ravel()
    result[6912:7296] = (
        original[6912:7296]
        + ((mean.astype(np.float64) - encoder["mean"]) / encoder["std"]) @ weights
    )
    return result


def initial_theta(variant, arm, encoders, mean, std):
    """Rebuild every initial parameter independently of either training Bank class."""
    assert variant in PARAMETERS and arm in ARMS
    layers = [("token1", 18, 384), ("token2", 384, 256)]
    if variant == "patch5m":
        layers += [
            ("mlp1", 292, 1792),
            ("mlp2", 1792, 1536),
            ("mlp3", 1536, 1024),
            ("head", 1024, 3),
        ]
    else:
        layers += [
            ("context", 292, 1120),
            ("query", 1123, 256),
            ("key", 256, 256),
            ("update1", 2499, 1120),
            ("update2", 1120, 1120),
            ("head", 1120, 3),
        ]
    output = np.empty((6, PARAMETERS[variant] - 643), np.float32)
    offset = 0
    for name, inputs, outputs in layers:
        size = (inputs + (name != "key")) * outputs
        for si, seed in enumerate(SEEDS):
            if name == "head":
                values = np.zeros(size, np.float32)
            else:
                generator = torch.Generator().manual_seed(seed + zlib.crc32(name.encode()))
                values = (
                    torch.empty(size)
                    .uniform_(-1 / math.sqrt(inputs), 1 / math.sqrt(inputs), generator=generator)
                    .numpy()
                )
            output[2 * si : 2 * si + 2, offset : offset + size] = values
        offset += size
    assert offset == output.shape[1]
    if arm == "original":
        assert encoders is None
    else:
        assert len(encoders) == 3
        for si, seed in enumerate(SEEDS):
            validate_encoder(encoders[si], seed, arm)
            output[2 * si : 2 * si + 2, :ENCODER_PARAMETERS] = transferred(encoders[si], mean, std)
    return output


def selected_heads(selection):
    result = {}
    for role in ROLES:
        result[role] = {}
        for variant in PARAMETERS:
            current = selection["roles"][role]
            expected = rank([c for c in current["candidates"] if c.get("architecture") == variant])
            choice = current["policies"]["per_architecture"][variant]
            assert choice == expected
            mode = choice["head_mode"]
            assert mode in ("unit", "wide", "linear") and choice["variant"] == variant + "__" + mode
            result[role][variant] = mode
    return result


def select_policies(candidates):
    overall = rank(candidates)
    return dict(
        per_pair={
            v + "__" + a: rank([c for c in candidates if c["variant"] == v + "__" + a])
            for v in PARAMETERS
            for a in ARMS
        },
        per_architecture={
            v: rank([c for c in candidates if c.get("architecture") == v]) for v in PARAMETERS
        },
        overall=overall,
    )


def payload(record):
    while "model" not in record:
        if "inherited_hr_record" in record:
            record = record["inherited_hr_record"]
        elif "inherited_as_record" in record:
            record = record["inherited_as_record"]
        else:
            raise ValueError("No exported model in inherited record lineage")
    assert "output" in record
    return record


def native_costs(banks, run_seconds):
    assert math.isfinite(run_seconds) and run_seconds >= 0
    for bank in banks:
        assert bank["steps"] > 0
        assert (
            0
            <= bank["setup_seconds"]
            <= bank["full_bank_seconds"]
            <= bank["write_and_prediction_inclusive_seconds"]
        )
    return dict(
        native_banks=len(banks),
        native_trajectories=6 * len(banks),
        native_bank_seconds=sum(b["full_bank_seconds"] for b in banks),
        native_setup_seconds=sum(b["setup_seconds"] for b in banks),
        native_write_inclusive_seconds=sum(
            b["write_and_prediction_inclusive_seconds"] for b in banks
        ),
        presentations=64 * 6 * sum(b["steps"] for b in banks),
        peak_allocated_cuda_bytes=max(b["cuda_peak_allocated_bytes"] for b in banks),
        primary_runtime_seconds=run_seconds,
        complete_end_to_end_seconds=None,
        scope="Overlapping fit/setup/write timings; primary timer excludes preparation. Never divide banks by six.",
    )


def primary_gate():
    require_terminal(read(HR_RUN / "job.json"))
    if not (RUN / "job.json").exists():
        raise RuntimeError("P3 native primary has not run")
    try:
        require_terminal(read(RUN / "job.json"))
    except RuntimeError as exc:
        raise RuntimeError("P3 primary must be complete and its worker confirmed dead") from exc


def registration_check():
    assert digest(RUN / "registration.json") == REGISTRATION
    registration = read(RUN / "registration.json")
    check_hashes(registration["bindings"])
    assert digest(HR_RUN / "source_lock.json") == HR_SOURCE
    assert read(HR_RUN / "source_lock.json")["cache_sha256"] == NATIVE_CACHE_SHA
    check_hashes(NATIVE_INPUTS)
    return registration


def freeze_contract():
    registration_check()
    assert not (RUN / "job.json").exists() and not (OUT / "verification.json").exists()
    value = dict(
        registration_sha256=REGISTRATION,
        sources={f: digest(ROOT / f) for f in FILES},
        auxiliary_cost_bindings={
            str(P2 / name): digest(P2 / name)
            for name in ("data_profile.json", "fit.json", "data_audit.json", "fit_audit.json")
        },
        native_input_bindings=NATIVE_INPUTS,
        expected_counts=EXPECTED,
        runtime_counts=RUNTIME_COUNTS,
        atol_native_lab=0.002,
        rtol=1e-6,
        requirements="Full registered P3 audit, all-pass single calls, actual CPU responses and complete native replay; auxiliary costs separate.",
        classification="verification consumers prepared; no native P3 fitting or quality outcome implied",
    )
    write_once(CONTRACT, value)
    return value


def verify_contract(production=True):
    registration_check()
    value = read(CONTRACT)
    assert value["registration_sha256"] == REGISTRATION
    assert value["expected_counts"] == EXPECTED and value["runtime_counts"] == RUNTIME_COUNTS
    assert value["native_input_bindings"] == NATIVE_INPUTS
    check_hashes(
        {**value["sources"], **value["auxiliary_cost_bindings"], **value["native_input_bindings"]}
    )
    if production:
        primary_gate()
        lock = read(RUN / "source_lock.json")
        source = digest(RUN / "source_lock.json")
        assert lock["registration_sha256"] == REGISTRATION
        assert lock["hr_verification_sha256"] == digest(HR_OUT / "verification.json")
        seal = read(HR_OUT / "verification.json")
        assert seal["passed"] and seal["source_lock_sha256"] == HR_SOURCE
        assert seal["selection_sha256"] == digest(HR_RUN / "selections.json")
        assert seal["results_sha256"] == digest(HR_RUN / "results.json")
        check_hashes(
            {
                **seal["sources"],
                **seal["inputs"],
                **seal["postprocess_sources"],
                **seal["artifact_sha256"],
                **lock["bindings"],
            }
        )
        assert lock["heads"] == selected_heads(read(HR_RUN / "selections.json"))
        for device in ("cpu", "cuda"):
            gate = read(RUN / f"preflight_{device}.json")
            assert gate["passed"] and gate["registration_sha256"] == REGISTRATION
            assert (
                gate["original_exact_payloads"] == gate["transferred_prefix_exact_payloads"] == 108
            )
        job, selection, results = (
            read(RUN / name) for name in ("job.json", "selections.json", "results.json")
        )
        assert (
            job["source_lock_sha256"]
            == selection["source_lock_sha256"]
            == results["source_lock_sha256"]
            == source
        )
        assert results["selection_sha256"] == digest(RUN / "selections.json")
        assert job["results_sha256"] == digest(RUN / "results.json")
        # A future recovery needs its own bound cost adapter, not an assumed zero overhead.
        if any(RUN.glob("recovery*")):
            raise RuntimeError(
                "Preserved P3 recovery requires a separately bound verifier and cost accounting"
            )
    return value


def verify_stage(name):
    value = read(OUT / name)
    assert value["passed"] and value["verification_protocol_sha256"] == digest(CONTRACT)
    assert value["results_sha256"] == digest(RUN / "results.json")
    assert value["dependencies"] == read(CONTRACT)["sources"]
    check_hashes({**value["dependencies"], **value["artifact_sha256"]})
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "verify", "guard"))
    action = parser.parse_args().action
    if action == "freeze":
        freeze_contract()
        print("P3 VERIFICATION CONSUMERS FROZEN", digest(CONTRACT), flush=True)
    elif action == "guard":
        primary_gate()
        print("P3 terminal worker confirmed dead", flush=True)
    else:
        verify_contract(production=False)
        print("P3 verification sources unchanged", flush=True)


if __name__ == "__main__":
    main()
