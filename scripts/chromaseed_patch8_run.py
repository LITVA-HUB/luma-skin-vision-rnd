"""Nested P8 local-branch experiment with fixed bank shape and source lineage."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import CACHE, ROOT, context, save_npz
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_neural_prefix_run import check_map
from chromaseed_patch8_fit import ARMS, HORIZON, RATES, SLOTS, Bank, fit, token_normalizers
from chromaseed_patch8_numpy import EXTRA, choose, predict, transform_tokens
from chromaseed_refine_train import setup
from chromaseed_weight_average_run import OUT as WA_OUT
from chromaseed_weight_average_run import RUN as WA
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json

RUN = ROOT / "experiments/runs/chromaseed_patch8_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_patch8_v1"
CHECKPOINTS = (0, 512, 2048, 8192, 32768)
PARENT = "f82f7f7d31b5639abbfef57ce96c79709022205b5d583291cf932cd823b1bdc0"


def load_data():
    assert sha(CACHE) == CACHE_HASH
    with np.load(CACHE, allow_pickle=False) as z:
        d = {k: z[k] for k in ("color", "tokens", "target", "patient", "site", "device")}
    assert (
        d["color"].shape == (966, 36)
        and d["tokens"].shape == (966, 64, 18)
        and d["target"].shape == (966, 3)
    )
    assert len(np.unique(d["patient"])) == 24 and np.isfinite(d["tokens"]).all()
    return d


def bank_path(role, fold=None):
    return (
        RUN
        / ("final" if fold is None else "inner")
        / role
        / ("bank" if fold is None else f"fold{fold}")
    )


def slot_index(si, arm, lr):
    return si * 6 + ARMS.index(arm) * 3 + (0 if lr is None else RATES.index(lr))


def freeze():
    assert sha(WA_OUT / "verification.json") == PARENT
    old, seal = js(WA / "source_lock.json"), js(WA_OUT / "verification.json")
    sources = {**old["sources"], **seal["postprocess_sources"]}
    inputs = {**old["input_sha256"], **seal["artifact_sha256"]}
    inputs[(WA_OUT / "verification.json").relative_to(ROOT).as_posix()] = PARENT
    for p in (
        "scripts/chromaseed_patch8_numpy.py",
        "scripts/chromaseed_patch8_fit.py",
        "scripts/chromaseed_patch8_run.py",
        "tests/test_chromaseed_patch8.py",
        "docs/research/chromaseed_patch8_v1_protocol.md",
        "scripts/skin_mskcc_pixels.py",
        "scripts/chromaseed_refine.py",
        "docs/research/chromaseed_refine_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    p = "docs/benchmarks/chromaseed_refine_v1/summary.json"
    inputs[p] = sha(ROOT / p)
    check_map({**sources, **inputs})
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    value = dict(
        sources=sources,
        input_sha256=inputs,
        parent_verification_sha256=PARENT,
        cache_sha256=CACHE_HASH,
        slots=list(SLOTS),
        horizon=HORIZON,
        checkpoints=list(CHECKPOINTS),
        inner_banks=9,
        final_banks=3,
        actual_trajectories=216,
        serialized_checkpoints=1080,
        candidates=78,
        policies=9,
        previous_turn_classification="progress",
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        numpy=np.__version__,
    )
    path = RUN / "source_lock.json"
    if path.exists():
        assert js(path) == value, "P8 sources/parents changed after freeze"
    else:
        write_json(path, value)
    return sha(path)


def train_one(data, source, role, fold=None):
    ix, query, warm, parents = context(data, role, fold)
    path = bank_path(role, fold)
    selector = None if fold is not None else sha(RUN / "selections.json")
    if (path / "receipt.json").exists():
        old = js(path / "receipt.json")
        assert (
            old["source_lock_sha256"] == source
            and old["selection_sha256"] == selector
            and old["warm_parent_sha256"] == parents
        )
        for name, digest in old["files"].items():
            assert sha(path / name) == digest
        print("P8 REUSE", role, fold, flush=True)
        return
    print("P8 START", role, fold, "rows", len(ix), flush=True)

    def progress(item):
        if item["step"] in CHECKPOINTS or item["step"] % 8192 == 0:
            print("P8 TRAIN", role, fold, item["step"], round(item["seconds"], 2), flush=True)

    w = weights_for(data["patient"][ix], data["site"][ix])
    models, info = fit(
        data["color"][ix],
        data["tokens"][ix],
        data["target"][ix],
        w,
        warm,
        HORIZON,
        CHECKPOINTS,
        "cuda",
        "cuda_graph",
        progress,
    )
    files = {}
    p = path / "warm_models.npz"
    save_npz(p, flatten({str(i): v for i, v in enumerate(warm)}))
    files[p.name] = sha(p)
    for step, payloads in models.items():
        p = path / f"models_{step}.npz"
        save_npz(p, flatten({str(i): v for i, v in enumerate(payloads)}))
        files[p.name] = sha(p)
        if fold is not None:
            p = path / f"oof_{step}.npz"
            output = np.stack(
                [predict(v, data["color"][query], data["tokens"][query]) for v in payloads]
            )
            save_npz(p, dict(row_indices=query, predictions=output))
            files[p.name] = sha(p)
    p = path / "rows.npz"
    save_npz(p, dict(fit_rows=ix, query_rows=query if fold is not None else np.array([], np.int64)))
    files[p.name] = sha(p)
    info.update(
        source_lock_sha256=source,
        selection_sha256=selector,
        role=role,
        fold=fold,
        warm_parent_sha256=parents,
        files=files,
    )
    write_json(path / "receipt.json", info)
    print("P8 SAVED", role, fold, flush=True)


def select(data, source):
    result = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        values = {}
        for step in CHECKPOINTS:
            parts = [nz(bank_path(role, f) / f"oof_{step}.npz") for f in range(3)]
            rows = np.concatenate([p["row_indices"] for p in parts])
            order = np.argsort(rows)
            np.testing.assert_array_equal(rows[order], ix)
            values[step] = np.concatenate([p["predictions"] for p in parts], axis=1)[:, order]
        candidates = []
        for arm in ARMS:
            for step in CHECKPOINTS:
                for lr in (None,) if step == 0 else RATES:
                    mm = [
                        metrics(
                            values[step][slot_index(si, arm, lr)],
                            data["target"][ix],
                            data["patient"][ix],
                            data["site"][ix],
                        )
                        for si in range(3)
                    ]
                    candidates.append(
                        dict(
                            arm=arm,
                            step=step,
                            lr=lr,
                            numeric_bytes=2886 if arm == "stats" else 5174,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
        assert len(candidates) == 26
        policies = {a: choose([c for c in candidates if c["arm"] == a]) for a in ARMS}
        policies["overall"] = choose(candidates)
        result["roles"][role] = dict(candidates=candidates, policies=policies)
    path = RUN / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    print("P8 SELECTION FROZEN", sha(path), flush=True)
    return result


def evaluate(data, source, selection):
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        query = np.flatnonzero(held)
        for step in CHECKPOINTS:
            packed = nz(bank_path(role) / f"models_{step}.npz")
            for slot, hp in enumerate(SLOTS):
                model = unpack(packed, str(slot))
                li = RATES.index(hp["lr"])
                name = f"{hp['arm']}_r{li}_t{step}_s{hp['seed']}"
                p, pp = (
                    RUN / "selected" / role / f"{name}.npz",
                    RUN / "evaluated" / role / f"{name}.npz",
                )
                save_npz(p, model)
                output = np.stack(
                    [
                        predict(
                            model,
                            affine_features(data["color"][query], s["dose"], s["anchor"]),
                            transform_tokens(data["tokens"][query], s["dose"], s["anchor"]),
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
                permuted = predict(model, data["color"][query], data["tokens"][query, ::-1].copy())
                permutation = float(np.max(abs(permuted - output[0])))
                assert permutation <= 2e-8
                ablated = base_predict(
                    {k: v for k, v in model.items() if k not in EXTRA}, data["color"][query]
                )
                selected = [
                    policy
                    for policy, c in selection["roles"][role]["policies"].items()
                    if c["arm"] == hp["arm"]
                    and c["step"] == step
                    and (li == 0 if step == 0 else c["lr"] == hp["lr"])
                ]
                records.append(
                    dict(
                        role=role,
                        name=name,
                        slot=slot,
                        **hp,
                        step=step,
                        baseline_alias=step == 0,
                        policies=selected,
                        parameters=643 if hp["arm"] == "stats" else 1179,
                        numeric_bytes=sum(
                            v.nbytes for v in model.values() if v.dtype.kind in "biufc"
                        ),
                        archive_bytes=p.stat().st_size,
                        model_sha256=sha(p),
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
                        permutation_max_lab=permutation,
                        branch_ablation_metrics=metrics(
                            ablated,
                            data["target"][query],
                            data["patient"][query],
                            data["site"][query],
                            data["device"][query],
                        ),
                        branch_ablation_max_lab=float(np.max(abs(output[0] - ablated))),
                    )
                )
            print("P8 EVALUATED", role, step, flush=True)
    assert len(records) == 270 and sum("overall" in r["policies"] for r in records) == 9
    value = dict(
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        records=records,
        evidence="Matched small local branch on existing original TRAIN tokens; no ordinary-phone face validation",
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
    assert not (OUT / "verification.json").exists(), "sealed P8: report verifier only"
    setup("cuda")
    data = load_data()
    if args.stage == "preflight":
        assert not (RUN / "source_lock.json").exists()
        for role in roles(data["patient"], data["device"]):
            for fold in (0, 1, 2, None):
                ix, _, warm, _ = context(data, role, fold)
                net = Bank(warm)
                prep = token_normalizers(data["tokens"][ix])
                for slot in range(18):
                    m = net.export(slot, warm, prep)
                    np.testing.assert_array_equal(
                        predict(m, data["color"][ix[:2]], data["tokens"][ix[:2]]),
                        base_predict(warm[slot // 6], data["color"][ix[:2]]),
                    )
        print(
            "P8 PREFLIGHT passed12matching warm banks,216exactinitial mappings,1179/643params",
            flush=True,
        )
        return
    started = time.perf_counter()
    source = freeze()
    for role in roles(data["patient"], data["device"]):
        for fold in range(3):
            train_one(data, source, role, fold)
    selection = select(data, source)
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
    print("P8 COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
