"""Nested LT continuation: nine inner banks, frozen choices, three long final banks."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from chromaseed_fast_kernel_train import row_hash
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_train import CACHE, ROOT, load_data, save_npz
from chromaseed_local_denoise_train import RUN as ND
from chromaseed_long_training_fit import HORIZON, MODES, RATES, SEEDS, SLOTS, fit
from chromaseed_neural_prefix_numpy import export_prefix, predict
from chromaseed_neural_prefix_run import RUN as NP
from chromaseed_neural_prefix_run import check_map
from chromaseed_refine_train import setup
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)

RUN = ROOT / "experiments/runs/chromaseed_long_training_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_long_training_v1"
NB = ROOT / "experiments/runs/chromaseed_neural_blocks_v1"
NB_OUT = ROOT / "docs/benchmarks/chromaseed_neural_blocks_v1"
CHECKPOINTS = (0, 512, 2048, 8192, 32768, 131072)
PARENT_HASH = "671fb2cd20d5e914bbddf3f5f9abafd05a73ea6ecee57c244e769a1cd47f8f2c"


def context(data, role, fold=None):
    fitmask, held = roles(data["patient"], data["device"])[role]
    ix = np.flatnonzero(fitmask)
    prior = js(NP / "selections.json")["roles"][role]["blind4"]
    j = prior["policies"]["quality"]["prefix"]
    assert j == prior["policies"]["compact"]["prefix"]
    parents = {}
    if fold is None:
        query = np.flatnonzero(held)
        warm = []
        for seed in SEEDS:
            path = NP / "selected" / role / f"blind4_j{j}_s{seed}.npz"
            warm.append(nz(path))
            parents[path.relative_to(ROOT).as_posix()] = sha(path)
    else:
        assignment = folds_for(data["patient"][ix], data["device"][ix])
        query, ix = ix[assignment == fold], ix[assignment != fold]
        path = ND / "inner" / role / "blind4" / f"fold{fold}"
        oldrows = nz(path / "rows.npz")
        np.testing.assert_array_equal(ix, oldrows["fit_rows"])
        np.testing.assert_array_equal(query, oldrows["query_rows"])
        rawpath = path / f"models_{prior['step']}.npz"
        bank = nz(rawpath)
        warm = [export_prefix(unpack(bank, str(2 * si + prior["lr_index"])), j) for si in range(3)]
        parents[rawpath.relative_to(ROOT).as_posix()] = sha(rawpath)
        parents[(path / "rows.npz").relative_to(ROOT).as_posix()] = sha(path / "rows.npz")
    assert not set(data["patient"][ix]) & set(data["patient"][query])
    return ix, query, warm, parents


def freeze():
    assert sha(CACHE) == CACHE_HASH and sha(NB_OUT / "verification.json") == PARENT_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    inherited, parent = js(NB / "source_lock.json"), js(NB_OUT / "verification.json")
    sources = {**inherited["sources"], **parent["postprocess_sources"]}
    inputs = {**inherited["input_sha256"], **parent["artifact_sha256"]}
    check_map({**sources, **inputs})
    inputs[(NB_OUT / "verification.json").relative_to(ROOT).as_posix()] = PARENT_HASH
    for name in (
        "scripts/chromaseed_long_training_fit.py",
        "scripts/chromaseed_long_training_run.py",
        "tests/test_chromaseed_long_training.py",
        "docs/research/chromaseed_long_training_v1_protocol.md",
    ):
        sources[name] = sha(ROOT / name)
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        parent_verification_sha256=PARENT_HASH,
        slots=list(SLOTS),
        checkpoints=list(CHECKPOINTS),
        horizon=HORIZON,
        inner_banks=9,
        final_banks=3,
        candidates=99,
        choices=12,
        actual_trajectories=216,
        serialized_checkpoints=1296,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        numpy=np.__version__,
        previous_turn_classification="progress",
        user_steering="More examples and longer training for the light model; synthetic variants are not new measured people",
    )
    path = RUN / "source_lock.json"
    if path.exists():
        assert js(path) == value, "LT sources/inputs changed after freeze"
    else:
        write_json(path, value)
    return sha(path)


def bank_path(role, fold=None):
    return (
        RUN
        / ("final" if fold is None else "inner")
        / role
        / ("bank" if fold is None else f"fold{fold}")
    )


def train_one(data, source, role, fold=None):
    path = bank_path(role, fold)
    ix, query, warm, parents = context(data, role, fold)
    selection = None if fold is not None else sha(RUN / "selections.json")
    if (path / "receipt.json").exists():
        old = js(path / "receipt.json")
        assert old["source_lock_sha256"] == source and old["selection_sha256"] == selection
        assert old["fit_rows_sha256"] == row_hash(ix) and old["warm_parent_sha256"] == parents
        for name, digest in old["files"].items():
            assert sha(path / name) == digest
        print("LT REUSE", path.relative_to(RUN).as_posix(), flush=True)
        return
    print(
        "LT START",
        path.relative_to(RUN).as_posix(),
        "fit rows",
        len(ix),
        "pool",
        len(ix) * 257,
        flush=True,
    )

    def progress(entry):
        if entry["step"] in CHECKPOINTS or entry["step"] % 32768 == 0:
            print(
                "LT BANK",
                path.relative_to(RUN).as_posix(),
                entry["step"],
                round(entry["seconds"], 2),
                flush=True,
            )

    w = weights_for(data["patient"][ix], data["site"][ix])
    models, info = fit(
        data["color"][ix],
        data["target"][ix],
        w,
        ix,
        warm,
        HORIZON,
        CHECKPOINTS,
        "cuda",
        "cuda_graph",
        progress,
    )
    files = {}
    save_npz(path / "warm_models.npz", flatten({str(i): p for i, p in enumerate(warm)}))
    files["warm_models.npz"] = sha(path / "warm_models.npz")
    for step, payloads in models.items():
        p = path / f"models_{step}.npz"
        save_npz(p, flatten({str(i): v for i, v in enumerate(payloads)}))
        files[p.name] = sha(p)
        if fold is not None:
            p = path / f"oof_{step}.npz"
            save_npz(
                p,
                dict(
                    row_indices=query,
                    predictions=np.stack([predict(v, data["color"][query]) for v in payloads]),
                ),
            )
            files[p.name] = sha(p)
    save_npz(
        path / "rows.npz",
        dict(fit_rows=ix, query_rows=query if fold is not None else np.array([], np.int64)),
    )
    files["rows.npz"] = sha(path / "rows.npz")
    info.update(
        role=role,
        fold=fold,
        source_lock_sha256=source,
        selection_sha256=selection,
        fit_rows_sha256=row_hash(ix),
        warm_parent_sha256=parents,
        files=files,
    )
    write_json(path / "receipt.json", info)
    print("LT SAVED", path.relative_to(RUN).as_posix(), flush=True)


def slot_index(seed_index, variants, lr):
    return seed_index * 6 + MODES.index(variants) * 2 + (0 if lr is None else RATES.index(lr))


def choose(candidates, across_modes=False):
    return min(
        candidates,
        key=lambda c: (
            c["clean"],
            c["p90"],
            c["step"],
            c["variants"] if across_modes else 0,
            c["lr"] or 0,
        ),
    )


def select(data, source):
    result = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        candidate = []
        saved = {}
        for step in CHECKPOINTS:
            parts = [nz(bank_path(role, fold) / f"oof_{step}.npz") for fold in range(3)]
            rows = np.concatenate([p["row_indices"] for p in parts])
            order = np.argsort(rows)
            np.testing.assert_array_equal(rows[order], ix)
            saved[step] = np.concatenate([p["predictions"] for p in parts], axis=1)[:, order]
        for variants in MODES:
            for step in CHECKPOINTS:
                for lr in (None,) if step == 0 else RATES:
                    mm = [
                        metrics(
                            saved[step][slot_index(si, variants, lr)],
                            data["target"][ix],
                            data["patient"][ix],
                            data["site"][ix],
                        )
                        for si in range(3)
                    ]
                    candidate.append(
                        dict(
                            variants=variants,
                            step=step,
                            lr=lr,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
        assert len(candidate) == 33
        result["roles"][role] = dict(
            candidates=candidate,
            per_mode={str(v): choose([c for c in candidate if c["variants"] == v]) for v in MODES},
            overall=choose(candidate, True),
        )
    path = RUN / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    return result


def evaluate(data, source, selection):
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        query = np.flatnonzero(held)
        policies = selection["roles"][role]
        for step in CHECKPOINTS:
            bank = nz(bank_path(role) / f"models_{step}.npz")
            for i, slot in enumerate(SLOTS):
                model = unpack(bank, str(i))
                li = RATES.index(slot["lr"])
                name = f"v{slot['variants']}_r{li}_t{step}_s{slot['seed']}"
                path, pp = (
                    RUN / "selected" / role / f"{name}.npz",
                    RUN / "evaluated" / role / f"{name}.npz",
                )
                save_npz(path, model)
                output = np.stack(
                    [
                        predict(
                            model, affine_features(data["color"][query], s["dose"], s["anchor"])
                        )
                        for s in grid()
                    ]
                )
                save_npz(pp, dict(row_indices=query, predictions=output))
                transforms, doses = summaries(
                    output,
                    data["target"][query],
                    data["patient"][query],
                    data["device"][query],
                    None,
                )

                def matches(c):
                    return (
                        step == c["step"]
                        and slot["variants"] == c["variants"]
                        and (li == 0 if step == 0 else slot["lr"] == c["lr"])
                    )

                records.append(
                    dict(
                        role=role,
                        name=name,
                        slot=i,
                        **slot,
                        step=step,
                        baseline_alias=step == 0,
                        selected_per_mode=matches(policies["per_mode"][str(slot["variants"])]),
                        selected_overall=matches(policies["overall"]),
                        parameters=643,
                        numeric_bytes=sum(
                            v.nbytes for v in model.values() if v.dtype.kind in "biufc"
                        ),
                        archive_bytes=path.stat().st_size,
                        model_sha256=sha(path),
                        prediction_sha256=sha(pp),
                        metrics=metrics(
                            output[0],
                            data["target"][query],
                            data["patient"][query],
                            data["site"][query],
                            data["device"][query],
                        ),
                        transforms=transforms,
                        doses=doses,
                    )
                )
            print("LT EVALUATED", role, step, flush=True)
    assert (
        len(records) == 324
        and sum(r["selected_per_mode"] for r in records) == 27
        and sum(r["selected_overall"] for r in records) == 9
    )
    value = dict(
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        records=records,
        evidence="Conditional continuation with synthetic color variants; no new measured people; repeated original TRAIN roles",
    )
    path = RUN / "results.json"
    if path.exists():
        assert js(path) == value
    else:
        write_json(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "all"), default="all")
    args = parser.parse_args()
    assert not (OUT / "verification.json").exists(), "sealed LT: report verifier only"
    setup("cuda")
    data = load_data()
    if args.stage == "preflight":
        assert not (RUN / "source_lock.json").exists()
        for role in roles(data["patient"], data["device"]):
            for fold in (0, 1, 2, None):
                ix, query, warm, _ = context(data, role, fold)
                assert len(warm) == 3 and len(ix) > 0 and len(query) > 0
        print(
            dict(
                stage="preflight",
                banks=12,
                trajectories=216,
                slots=18,
                steps=HORIZON,
                checkpoints=CHECKPOINTS,
                synthetic_variants=MODES,
            )
        )
        return
    started = time.perf_counter()
    source = freeze()
    for role in roles(data["patient"], data["device"]):
        for fold in range(3):
            train_one(data, source, role, fold)
    selection = select(data, source)
    print("LT SELECTION FROZEN", sha(RUN / "selections.json"), flush=True)
    for role in roles(data["patient"], data["device"]):
        train_one(data, source, role)
    evaluate(data, source, selection)
    write_json(
        RUN / "progress.json",
        dict(
            status="complete",
            pid=os.getpid(),
            seconds=time.perf_counter() - started,
            source_lock_sha256=source,
            results_sha256=sha(RUN / "results.json"),
        ),
    )
    print("LT COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
