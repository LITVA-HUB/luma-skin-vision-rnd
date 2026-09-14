"""P3 preregistration and matched native run, gated on completed and sealed HR."""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from chromaseed_architecture_scale import BATCH, HORIZON, RATES, SEEDS, SLOTS, capacity
from chromaseed_gated import flatten, unpack
from chromaseed_head_range import predict_torch
from chromaseed_head_range_audit import bank_path as hr_bank
from chromaseed_head_range_verification import (
    CONTRACT as HR_CONTRACT,
)
from chromaseed_head_range_verification import (
    OUT as HR_OUT,
)
from chromaseed_head_range_verification import (
    ROLES,
    ROOT,
    check_hashes,
    digest,
    read,
    require_quiet_host,
    require_terminal,
    write_once,
)
from chromaseed_head_range_verification import (
    RUN as HR_RUN,
)
from chromaseed_head_range_verification import (
    SOURCE as HR_SOURCE,
)
from chromaseed_kernel_audit import nz
from chromaseed_long_training_run import context
from chromaseed_palette_transfer import ARMS, VARIANTS, validate_encoders
from chromaseed_palette_transfer_fit import fit
from chromaseed_patch8_numpy import choose
from chromaseed_refine_train import setup
from chromaseed_widen_run import load_data
from skin_local_search_train import metrics, roles, weights_for, write_json

RUN = Path("D:/Luma-RnD/chromaseed_palette_transfer_v1")
OUT = ROOT / "docs/benchmarks/chromaseed_palette_transfer_v1"
P2 = Path("D:/Luma-RnD/chromaseed_palette_pretrain_v1")
TIMES = (128, 512, 2048)
FILES = (
    "scripts/chromaseed_palette_transfer.py",
    "scripts/chromaseed_palette_transfer_fit.py",
    "scripts/chromaseed_palette_transfer_run.py",
    "scripts/chromaseed_palette_transfer_preflight.py",
    "tests/test_chromaseed_palette_transfer.py",
    "docs/research/chromaseed_palette_transfer_v1_protocol.md",
)


def palette_bindings():
    pins = {
        "source_lock.json": "db5e5c8d9eef8f7a066a0fac8296956507ceb59522b8ffb95b7f86ab8b5b2cf7",
        "fit.json": "37236660053b4534b7e5fd6d73df306c604bc4a5457aafc4dd12eb79bbae3328",
        "fit_audit.json": "6f05d64cfda7077b87bb3ff9ef5eb9d2ad227d2eebc4c927a46f4d4987687c22",
    }
    for name, sha in pins.items():
        assert digest(P2 / name) == sha
    bindings = dict(read(P2 / "source_lock.json")["bindings"])
    fitted = read(P2 / "fit.json")
    for name in (*pins, "fit_audit_protocol.json", "data_audit_protocol.json", "data_audit.json"):
        bindings[str(P2 / name)] = digest(P2 / name)
    for name, key in (("data.npz", "data_sha256"), ("aux_bank_2048.npz", "auxiliary_bank_sha256")):
        bindings[str(P2 / name)] = fitted[key]
    assert {(r["seed"], r["arm"]) for r in fitted["exports"]} == {
        (s, a) for s in SEEDS for a in ARMS[1:]
    }
    for row in fitted["exports"]:
        bindings[row["path"]] = row["sha256"]
    for name, source in (
        ("fit_audit_protocol.json", "chromaseed_palette_fit_audit.py"),
        ("data_audit_protocol.json", "chromaseed_palette_data_audit.py"),
    ):
        bindings[str(ROOT / "scripts" / source)] = read(P2 / name)["source_sha256"]
    assert read(P2 / "fit_audit.json")["passed"]
    return bindings


def encoders_for(arm):
    if arm == "original":
        return None
    rows = read(P2 / "fit.json")["exports"]
    encoders = []
    for seed in SEEDS:
        row = next(r for r in rows if r["seed"] == seed and r["arm"] == arm)
        assert digest(row["path"]) == row["sha256"]
        encoders.append(nz(row["path"]))
    validate_encoders(encoders, arm)
    return encoders


def resolve_heads(selection):
    result = {}
    for role in ROLES:
        result[role] = {}
        for variant in VARIANTS:
            choice = selection["roles"][role]["policies"]["per_architecture"][variant]
            mode = choice["head_mode"]
            if (
                mode not in ("unit", "wide", "linear")
                or choice["architecture"] != variant
                or choice["variant"] != variant + "__" + mode
            ):
                raise ValueError("HR inner-selected head has mismatched architecture metadata")
            result[role][variant] = mode
    return result


def register():
    assert digest(HR_RUN / "source_lock.json") == HR_SOURCE
    hr = read(HR_RUN / "source_lock.json")
    bindings = {**hr["sources"], **read(HR_CONTRACT)["sources"], **palette_bindings()}
    bindings.update({name: digest(ROOT / name) for name in FILES})
    bindings[str(HR_RUN / "source_lock.json")] = HR_SOURCE
    bindings[str(HR_CONTRACT)] = digest(HR_CONTRACT)
    check_hashes(bindings)
    value = dict(
        bindings=bindings,
        variants=list(VARIANTS),
        arms=list(ARMS),
        slots=[list(s) for s in SLOTS],
        checkpoints=list(TIMES),
        horizon=HORIZON,
        batch_size=BATCH,
        head_rule="completed HR INNER per_architecture head_mode, fixed across P3 arms",
        new_inner_banks=54,
        new_final_banks=18,
        new_trajectories=432,
        candidates=168,
        choices=39,
        final_records=99,
        classification="registered before HR outcomes; no new native quality result",
    )
    write_once(RUN / "registration.json", value)
    return value


def verify_registration():
    value = read(RUN / "registration.json")
    check_hashes(value["bindings"])
    return value


def require_hr_sealed():
    require_terminal(read(HR_RUN / "job.json"))
    seal = read(HR_OUT / "verification.json")
    assert seal["passed"] and seal["source_lock_sha256"] == HR_SOURCE
    assert seal["results_sha256"] == digest(HR_RUN / "results.json")
    assert seal["selection_sha256"] == digest(HR_RUN / "selections.json")
    check_hashes(
        {
            **seal["sources"],
            **seal["inputs"],
            **seal["postprocess_sources"],
            **seal["artifact_sha256"],
        }
    )
    return seal


def freeze():
    registration = verify_registration()
    parent = require_hr_sealed()
    probe = read(RUN / "preflight_cuda.json")
    cpu_probe = read(RUN / "preflight_cpu.json")
    assert cpu_probe["passed"] and cpu_probe["registration_sha256"] == digest(
        RUN / "registration.json"
    )
    assert probe["passed"] and probe["registration_sha256"] == digest(RUN / "registration.json")
    assert (
        probe["original_exact_payloads"] == 108
        and probe["transferred_prefix_exact_payloads"] == 108
    )
    bindings = {**registration["bindings"], **parent["artifact_sha256"]}
    bindings.update(
        {
            str(p): digest(p)
            for p in (
                HR_OUT / "verification.json",
                RUN / "registration.json",
                RUN / "preflight_cpu.json",
                RUN / "preflight_cuda.json",
            )
        }
    )
    check_hashes(bindings)
    heads = resolve_heads(read(HR_RUN / "selections.json"))
    value = dict(
        registration_sha256=digest(RUN / "registration.json"),
        bindings=bindings,
        heads=heads,
        hr_verification_sha256=digest(HR_OUT / "verification.json"),
        torch=torch.__version__,
        numpy=np.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    write_once(RUN / "source_lock.json", value)
    return digest(RUN / "source_lock.json"), heads


def bank_path(role, variant, arm, fold=None):
    return (
        RUN
        / ("final" if fold is None else "inner")
        / role
        / (variant + "__" + arm)
        / ("bank" if fold is None else f"fold{fold}")
    )


def mutable_json(path, value, telemetry=False):
    for delay in (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.5, 0.5):
        try:
            write_json(path, value)
            return
        except PermissionError:
            time.sleep(delay)
    if not telemetry:
        write_json(path, value)  # A required write remains fatal.
    else:
        print("P3 TELEMETRY temporarily skipped", str(path), flush=True)


def save_arrays(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **value)


def train_bank(data, source, role, variant, mode, arm, completed, fold=None, steps=2048):
    ix, query, warm, parents = context(data, role, fold)
    path = bank_path(role, variant, arm, fold)
    selector = None if fold is not None else digest(RUN / "selections.json")
    if (path / "receipt.json").exists():
        receipt = read(path / "receipt.json")
        assert receipt["source_lock_sha256"] == source and receipt["selection_sha256"] == selector
        assert receipt["warm_parent_sha256"] == parents and receipt["head_mode"] == mode
        assert receipt["initialization"] == arm and receipt["steps"] == steps
        for name, sha in receipt["files"].items():
            assert digest(path / name) == sha
        print("P3 REUSE", role, variant, arm, fold, flush=True)
        return
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Preserve interrupted partial bank before retry: {path}")
    assert shutil.disk_usage(RUN).free > 15_000_000_000
    prior = read(hr_bank(role, variant, mode, fold) / "receipt.json")
    assert prior["warm_parent_sha256"] == parents
    rows = nz(hr_bank(role, variant, mode, fold) / "rows.npz")
    np.testing.assert_array_equal(rows["fit_rows"], ix)
    if fold is not None:
        np.testing.assert_array_equal(rows["query_rows"], query)
    began = time.perf_counter()
    base = dict(
        status="training",
        pid=os.getpid(),
        stage="inner" if fold is not None else "final",
        role=role,
        variant=variant,
        head_mode=mode,
        initialization=arm,
        fold=fold,
        completed_banks=completed,
        total_banks=72,
        target_steps=steps,
        parameters=capacity(variant),
    )
    mutable_json(RUN / "progress.json", dict(base, step=0, seconds=0), telemetry=True)

    def progress(item):
        mutable_json(RUN / "progress.json", dict(base, **item), telemetry=True)
        print(
            "P3 TRAIN",
            role,
            variant,
            mode,
            arm,
            fold,
            item["step"],
            round(item["seconds"], 2),
            flush=True,
        )

    checkpoints = TIMES if fold is not None else (steps,)
    fitted, info = fit(
        data["color"][ix],
        data["tokens"][ix],
        data["target"][ix],
        weights_for(data["patient"][ix], data["site"][ix]),
        warm,
        variant,
        mode,
        arm,
        encoders_for(arm),
        steps,
        checkpoints,
        "cuda",
        "cuda_graph",
        progress,
    )
    files = {}

    def write(name, payload):
        save_arrays(path / name, payload)
        files[name] = digest(path / name)

    write("warm_models.npz", flatten({str(i): m for i, m in enumerate(warm)}))
    for step, models in fitted.items():
        write(f"models_{step}.npz", flatten({str(i): m for i, m in enumerate(models)}))
        if fold is not None:
            pred = np.stack(
                [predict_torch(m, data["color"][query], data["tokens"][query]) for m in models]
            )
            write(f"oof_{step}.npz", dict(row_indices=query, predictions=pred))
    write(
        "rows.npz",
        dict(fit_rows=ix, query_rows=query if fold is not None else np.array([], np.int64)),
    )
    info.update(
        source_lock_sha256=source,
        selection_sha256=selector,
        role=role,
        fold=fold,
        warm_parent_sha256=parents,
        files=files,
        write_and_prediction_inclusive_seconds=time.perf_counter() - began,
    )
    write_once(path / "receipt.json", info)
    print("P3 SAVED", role, variant, arm, fold, flush=True)


def policies(candidates):
    return dict(
        per_pair={
            v + "__" + a: choose([c for c in candidates if c["variant"] == v + "__" + a])
            for v in VARIANTS
            for a in ARMS
        },
        per_architecture={
            v: choose([c for c in candidates if c.get("architecture") == v]) for v in VARIANTS
        },
        overall=choose(candidates),
    )


def select(data, source, heads):
    previous = read(HR_RUN / "selections.json")
    value = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix, candidates = np.flatnonzero(mask), []
        for variant in VARIANTS:
            mode = heads[role][variant]
            for arm in ARMS:
                pair = variant + "__" + arm
                if arm == "original":
                    old = [
                        c
                        for c in previous["roles"][role]["candidates"]
                        if c["variant"] == variant + "__" + mode
                    ]
                    assert len(old) == 6
                    candidates.extend(dict(c, variant=pair, initialization=arm) for c in old)
                    continue
                for ri, rate in enumerate(RATES):
                    for step in TIMES:
                        parts = []
                        for fold in range(3):
                            path = bank_path(role, variant, arm, fold)
                            assert (
                                digest(path / f"oof_{step}.npz")
                                == read(path / "receipt.json")["files"][f"oof_{step}.npz"]
                            )
                            parts.append(nz(path / f"oof_{step}.npz"))
                        rows = np.concatenate([p["row_indices"] for p in parts])
                        order = np.argsort(rows)
                        np.testing.assert_array_equal(rows[order], ix)
                        pred = np.concatenate([p["predictions"] for p in parts], axis=1)[:, order][
                            [2 * i + ri for i in range(3)]
                        ]
                        mm = [
                            metrics(p, data["target"][ix], data["patient"][ix], data["site"][ix])
                            for p in pred
                        ]
                        candidates.append(
                            dict(
                                variant=pair,
                                architecture=variant,
                                head_mode=mode,
                                initialization=arm,
                                step=step,
                                lr=rate,
                                parameters=capacity(variant),
                                numeric_bytes=4 * capacity(variant) + 458,
                                clean=float(np.mean([m["person_mean"] for m in mm])),
                                p90=float(np.mean([m["p90"] for m in mm])),
                                seed_metrics=mm,
                            )
                        )
        candidates.extend(
            next(c for c in previous["roles"][role]["candidates"] if c["variant"] == k)
            for k in ("np", "we")
        )
        assert len(candidates) == 56
        chosen = policies(candidates)
        for variant in VARIANTS:
            c, old = (
                chosen["per_pair"][variant + "__original"],
                previous["roles"][role]["policies"]["per_architecture"][variant],
            )
            assert (c["step"], c["lr"], c["clean"], c["head_mode"]) == (
                old["step"],
                old["lr"],
                old["clean"],
                old["head_mode"],
            )
        value["roles"][role] = dict(candidates=candidates, policies=chosen)
    write_once(RUN / "selections.json", value)
    print("P3 SELECTION FROZEN", digest(RUN / "selections.json"), flush=True)
    return value


def evaluate(data, source, selection, heads):
    previous, records = read(HR_RUN / "results.json"), []
    for role, (_, mask) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(mask)
        overall = selection["roles"][role]["policies"]["overall"]["variant"]
        for variant in VARIANTS:
            mode = heads[role][variant]
            for arm in ARMS:
                pair = variant + "__" + arm
                c = selection["roles"][role]["policies"]["per_pair"][pair]
                for si, seed in enumerate(SEEDS):
                    if arm == "original":
                        old = next(
                            r
                            for r in previous["records"]
                            if (r["role"], r["variant"], r["seed"])
                            == (role, variant + "__" + mode, seed)
                        )
                        records.append(
                            dict(
                                role=role,
                                variant=pair,
                                architecture=variant,
                                head_mode=mode,
                                initialization=arm,
                                seed=seed,
                                step=old["step"],
                                lr=old["lr"],
                                inherited_hr_record=old,
                                metrics=old["metrics"],
                                overall=overall == pair,
                            )
                        )
                        continue
                    path = bank_path(role, variant, arm) / f"models_{c['step']}.npz"
                    slot = 2 * si + RATES.index(c["lr"])
                    model = unpack(nz(path), str(slot))
                    dest, out = (
                        RUN / "models" / role / f"{pair}_s{seed}.npz",
                        RUN / "evaluated" / role / f"{pair}_s{seed}.npz",
                    )
                    save_arrays(dest, model)
                    passes = predict_torch(
                        model, data["color"][rows], data["tokens"][rows], all_passes=True
                    )
                    pred = passes[:, -1]
                    save_arrays(out, dict(row_indices=rows, predictions=pred, passes=passes))
                    records.append(
                        dict(
                            role=role,
                            variant=pair,
                            architecture=variant,
                            head_mode=mode,
                            initialization=arm,
                            seed=seed,
                            step=c["step"],
                            lr=c["lr"],
                            slot=slot,
                            source_path=str(path),
                            source_sha256=digest(path),
                            model=str(dest),
                            model_sha256=digest(dest),
                            output=str(out),
                            output_sha256=digest(out),
                            parameters=capacity(variant),
                            numeric_bytes=sum(
                                v.nbytes for v in model.values() if v.dtype.kind in "biufc"
                            ),
                            metrics=metrics(
                                pred,
                                data["target"][rows],
                                data["patient"][rows],
                                data["site"][rows],
                            ),
                            overall=overall == pair,
                        )
                    )
                print("P3 EVALUATED", role, pair, flush=True)
        for kind in ("np", "we"):
            for old in [
                r for r in previous["records"] if r["role"] == role and r["variant"] == kind
            ]:
                records.append(
                    dict(
                        role=role,
                        variant=kind,
                        architecture=kind,
                        initialization="control",
                        head_mode="control",
                        seed=old["seed"],
                        step=old["step"],
                        lr=old["lr"],
                        inherited_hr_record=old,
                        metrics=old["metrics"],
                        overall=overall == kind,
                    )
                )
    assert len(records) == 99 and sum(r["overall"] for r in records) == 9
    value = dict(
        source_lock_sha256=source, selection_sha256=digest(RUN / "selections.json"), records=records
    )
    write_once(RUN / "results.json", value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.register:
        register()
        print("P3 REGISTERED", digest(RUN / "registration.json"), flush=True)
        return
    require_hr_sealed()
    require_quiet_host()
    assert not (OUT / "verification.json").exists() and not (RUN / "results.json").exists()
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    setup("cuda")
    source, heads = freeze()
    if args.freeze_only:
        print("P3 PRODUCTION FROZEN", source, flush=True)
        return
    if (RUN / "job.json").exists():
        raise RuntimeError(
            "Preserve the existing attempt and its failure costs; recovery needs a separate bound adapter"
        )
    data, began, completed = load_data(), time.perf_counter(), 0
    mutable_json(
        RUN / "job.json", dict(status="running", pid=os.getpid(), source_lock_sha256=source)
    )
    try:
        for role in ROLES:
            for variant in VARIANTS:
                for arm in ARMS[1:]:
                    for fold in range(3):
                        train_bank(
                            data, source, role, variant, heads[role][variant], arm, completed, fold
                        )
                        completed += 1
        selection = select(data, source, heads)
        for role in ROLES:
            for variant in VARIANTS:
                for arm in ARMS[1:]:
                    step = selection["roles"][role]["policies"]["per_pair"][variant + "__" + arm][
                        "step"
                    ]
                    train_bank(
                        data,
                        source,
                        role,
                        variant,
                        heads[role][variant],
                        arm,
                        completed,
                        steps=step,
                    )
                    completed += 1
        mutable_json(
            RUN / "progress.json",
            dict(status="evaluating", pid=os.getpid(), completed_banks=72, total_banks=72),
            telemetry=True,
        )
        evaluate(data, source, selection, heads)
    except BaseException as exc:
        mutable_json(
            RUN / "job.json",
            dict(
                status="failed",
                pid=os.getpid(),
                error=type(exc).__name__ + ": " + str(exc),
                completed_banks=completed,
            ),
        )
        raise
    mutable_json(
        RUN / "job.json",
        dict(
            status="complete",
            pid=os.getpid(),
            source_lock_sha256=source,
            results_sha256=digest(RUN / "results.json"),
            seconds=time.perf_counter() - began,
        ),
    )
    mutable_json(
        RUN / "progress.json",
        dict(status="complete", completed_banks=72, total_banks=72),
        telemetry=True,
    )
    print("P3 PRIMARY COMPLETE", digest(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
