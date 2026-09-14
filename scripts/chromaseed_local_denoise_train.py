"""Frozen, resumable ND nested selection and target-free evaluation."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from chromaseed_fast_kernel_train import row_hash
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian_train import infer
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_local_denoise import SPECS, Bank
from chromaseed_local_denoise_fit import fit
from chromaseed_local_denoise_numpy import predict
from chromaseed_perceptual_audit import balanced
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

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_local_denoise_v1"
NS = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
CACHE = ROOT.parents[1] / "luma-skin-vision-rnd/data/processed/skin_mskcc_pixels_v1/train.npz"
SEEDS, LRS, CHECKPOINTS = (17, 29, 43), (0.001, 0.003), (512, 2048, 8192)
SLOTS = tuple((seed, lr) for seed in SEEDS for lr in LRS)
PARENT_HASH = "e725b4a6e86df7a51dd9825089ca39c218100bc3e31f765a585946353fb9cd6f"


def load_data():
    assert CACHE.name == "train.npz" and sha(CACHE) == CACHE_HASH
    with np.load(CACHE, allow_pickle=False) as z:
        return {key: z[key] for key in ("color", "target", "patient", "site", "device")}


def references():
    return [
        r
        for r in js(NS / "results.json")["records"]
        if r["group"] == "raw36"
        and (r["family"] == "fg_norm_static" or (r["family"] == "norm" and r["basis"] == "random"))
    ]


def freeze():
    assert sha(CACHE) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    parent = ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1/verification.json"
    assert sha(parent) == PARENT_HASH and js(parent)["passed"]
    inherited = js(NS / "source_lock.json")
    sources = dict(inherited["sources"])
    for p in (
        "scripts/chromaseed_local_denoise.py",
        "scripts/chromaseed_local_denoise_numpy.py",
        "scripts/chromaseed_local_denoise_fit.py",
        "scripts/chromaseed_local_denoise_train.py",
        "scripts/chromaseed_refine.py",
        "scripts/chromaseed_refine_train.py",
        "tests/test_chromaseed_local_denoise.py",
        "docs/research/chromaseed_local_denoise_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    for p, h in {
        **inherited["sources"],
        **inherited["input_sha256"],
        **js(parent)["artifact_sha256"],
    }.items():
        assert sha(ROOT / p) == h, p
    inputs = {
        p: sha(ROOT / p)
        for p in (
            "docs/benchmarks/chromaseed_neural_shrinkage_v1/verification.json",
            "experiments/runs/chromaseed_neural_shrinkage_v1/source_lock.json",
            "experiments/runs/chromaseed_neural_shrinkage_v1/results.json",
            "docs/research/chromaseed_refine_next_decision.md",
            "docs/research/chromaseed_neural_shrinkage_next_decision.md",
            "docs/research/chromaseed_forum_update_after_ns_2026-09-13.md",
        )
    }
    rr = references()
    assert len(rr) == 18
    for r in rr:
        path = NS / "selected" / r["role"] / f"{r['name']}.npz"
        assert sha(path) == r["model_sha256"]
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        families=SPECS,
        seeds=SEEDS,
        lrs=LRS,
        checkpoints=CHECKPOINTS,
        batch_size=64,
        device="cuda",
        engine="cuda_graph",
        dtype="float32",
        tf32=False,
        numpy=np.__version__,
        torch=torch.__version__,
        python=platform.python_version(),
        gpu=torch.cuda.get_device_name(),
        previous_turn_classification="progress",
    )
    # JSON normalization makes immutable reload insensitive to tuple/list syntax.
    import json

    value = json.loads(json.dumps(value))
    path = RUN / "source_lock.json"
    if path.exists():
        assert js(path) == value, "ND source or input changed after freeze"
    else:
        write_json(path, value)
    return sha(path)


def save_npz(path, arrays):
    if path.exists():
        old = nz(path)
        assert set(old) == set(arrays)
        for key in old:
            np.testing.assert_array_equal(old[key], arrays[key])
    else:
        atomic_npz(path, arrays)


def ready(path, source):
    receipt = path / "receipt.json"
    if not receipt.exists():
        return False
    value = js(receipt)
    assert value["source_lock_sha256"] == source
    for name, digest in value["files"].items():
        assert sha(path / name) == digest
    return True


def export_drift(payload, x):
    net = Bank(str(payload["family"]), [17]).cuda()
    with torch.no_grad():
        net.theta.copy_(torch.as_tensor(payload["theta"], device="cuda"))
        xn = (x.astype(np.float32) - payload["x_mean"]) / payload["x_std"]
        out = (
            net.rollout(torch.as_tensor(xn, device="cuda")[None])[0]
            .cpu()
            .numpy()
            .transpose(1, 0, 2)
        )
    out = out * payload["y_std"] + payload["y_mean"]
    drift = float(np.max(abs(out - predict(payload, x))))
    assert drift <= 0.002, drift
    return drift


def train_one(
    data, path, fit_rows, query, source, family, slots, steps, checkpoints, role, fold=None
):
    selection = None if fold is not None else sha(RUN / "selections.json")
    if ready(path, source):
        old = js(path / "receipt.json")
        assert old["selection_sha256"] == selection and old["fit_rows_sha256"] == row_hash(fit_rows)
        assert old["query_rows_sha256"] == (None if query is None else row_hash(query))
        print("REUSE", path.relative_to(RUN).as_posix(), flush=True)
        return
    name = path.relative_to(RUN).as_posix()
    print("START", name, flush=True)
    x, y = data["color"][fit_rows], data["target"][fit_rows]
    w = weights_for(data["patient"][fit_rows], data["site"][fit_rows])
    np.testing.assert_allclose(
        w, balanced(data["patient"][fit_rows], data["site"][fit_rows]), rtol=1e-12, atol=1e-12
    )
    if query is not None:
        assert not set(data["patient"][fit_rows]) & set(data["patient"][query])

    def progress(entry):
        write_json(
            RUN / "progress.json",
            dict(
                status="running",
                pid=os.getpid(),
                bank=name,
                step=entry["step"],
                seconds=entry["seconds"],
                updated_unix=time.time(),
            ),
        )
        if entry["step"] in checkpoints:
            print(name, entry["step"], round(entry["seconds"], 3), flush=True)

    checkpoints_payload, receipt = fit(
        x, y, w, family, slots, steps, checkpoints, "cuda", "cuda_graph", progress
    )
    files = {}
    max_drift = 0.0
    check_x = x if query is None else np.concatenate((x, data["color"][query]))
    for step, payloads in checkpoints_payload.items():
        modelpath = path / f"models_{step}.npz"
        save_npz(modelpath, flatten({str(i): p for i, p in enumerate(payloads)}))
        files[modelpath.name] = sha(modelpath)
        for p in payloads:
            max_drift = max(max_drift, export_drift(p, check_x))
        if query is not None:
            predpath = path / f"oof_{step}.npz"
            save_npz(
                predpath,
                dict(
                    row_indices=query,
                    stages=np.stack([predict(p, data["color"][query]) for p in payloads]),
                ),
            )
            files[predpath.name] = sha(predpath)
    rowpath = path / "rows.npz"
    save_npz(
        rowpath,
        dict(fit_rows=fit_rows, query_rows=np.array([], np.int64) if query is None else query),
    )
    files[rowpath.name] = sha(rowpath)
    receipt.update(
        source_lock_sha256=source,
        selection_sha256=selection,
        role=role,
        fold=fold,
        fit_rows_sha256=row_hash(fit_rows),
        query_rows_sha256=None if query is None else row_hash(query),
        files=files,
        max_torch_numpy_native_lab=max_drift,
    )
    write_json(path / "receipt.json", receipt)


def inner(data, source):
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        folds = folds_for(data["patient"][ix], data["device"][ix])
        for family in SPECS:
            for fold in range(3):
                train_one(
                    data,
                    RUN / "inner" / role / family / f"fold{fold}",
                    ix[folds != fold],
                    ix[folds == fold],
                    source,
                    family,
                    SLOTS,
                    8192,
                    CHECKPOINTS,
                    role,
                    fold,
                )


def oof(role, family, step):
    parts = [
        nz(RUN / "inner" / role / family / f"fold{fold}" / f"oof_{step}.npz") for fold in range(3)
    ]
    rows = np.concatenate([p["row_indices"] for p in parts])
    order = np.argsort(rows)
    return rows[order], np.concatenate([p["stages"] for p in parts], axis=1)[:, order]


def select(data, source):
    path = RUN / "selections.json"
    result = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        result["roles"][role] = {}
        for family in SPECS:
            candidates = []
            for step in CHECKPOINTS:
                rows, outputs = oof(role, family, step)
                np.testing.assert_array_equal(rows, np.flatnonzero(mask))
                for li, lr in enumerate(LRS):
                    mm = [
                        metrics(
                            outputs[2 * si + li, :, -1],
                            data["target"][rows],
                            data["patient"][rows],
                            data["site"][rows],
                        )
                        for si in range(3)
                    ]
                    candidates.append(
                        dict(
                            step=step,
                            lr=lr,
                            lr_index=li,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
            selected = min(
                candidates, key=lambda c: (c["clean"], c["p90"], c["step"], c["lr_index"])
            )
            result["roles"][role][family] = dict(selected=selected, candidates=candidates)
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    return result


def final(data, source, selection):
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        for family in SPECS:
            chosen = selection["roles"][role][family]["selected"]
            for seed in SEEDS:
                train_one(
                    data,
                    RUN / "final" / role / family / f"s{seed}",
                    ix,
                    None,
                    source,
                    family,
                    [(seed, chosen["lr"])],
                    chosen["step"],
                    (chosen["step"],),
                    role,
                )


def evaluate(data, source, selection):
    metadata = []
    for role, (mask, held) in roles(data["patient"], data["device"]).items():
        query = np.flatnonzero(held)
        models = {}
        for family in SPECS:
            chosen = selection["roles"][role][family]["selected"]
            for seed in SEEDS:
                path = RUN / "final" / role / family / f"s{seed}" / f"models_{chosen['step']}.npz"
                name = f"{family}_s{seed}"
                models[name] = (
                    unpack(nz(path), "0"),
                    dict(
                        name=name,
                        role=role,
                        family=family,
                        seed=seed,
                        step=chosen["step"],
                        lr=chosen["lr"],
                        parameters=SPECS[family][0]
                        * ((SPECS[family][1] + 1) * SPECS[family][2] + (SPECS[family][2] + 1) * 3),
                        origin="ND_fit",
                    ),
                )
        for r in references():
            if r["role"] != role:
                continue
            family = "fg_norm_static" if r["family"] == "fg_norm_static" else "random_head"
            name = f"{family}_s{r['seed']}"
            oldpath = NS / "selected" / role / f"{r['name']}.npz"
            models[name] = (
                nz(oldpath),
                dict(
                    name=name,
                    role=role,
                    family=family,
                    seed=r["seed"],
                    step=None,
                    lr=None,
                    parameters=None,
                    origin="exact_NS_reference",
                    ns_name=r["name"],
                    ns_model_sha256=sha(oldpath),
                ),
            )
        assert len(models) == 21
        for name, (model, record) in models.items():
            path = RUN / "selected" / role / f"{name}.npz"
            save_npz(path, model)
            stages = []
            for setting in grid():
                xt = affine_features(data["color"][query], setting["dose"], setting["anchor"])
                out = predict(model, xt) if "theta" in model else infer(model, xt)[:, None, :]
                stages.append(out)
            stages = np.stack(stages)
            predpath = RUN / "evaluated" / role / f"{name}.npz"
            save_npz(predpath, dict(row_indices=query, stages=stages))
            transforms, doses = summaries(
                stages[:, :, -1],
                data["target"][query],
                data["patient"][query],
                data["device"][query],
                None,
            )
            record.update(
                numeric_bytes=sum(v.nbytes for v in model.values() if v.dtype.kind in "biufc"),
                archive_bytes=path.stat().st_size,
                model_sha256=sha(path),
                prediction_sha256=sha(predpath),
                metrics=metrics(
                    stages[0, :, -1],
                    data["target"][query],
                    data["patient"][query],
                    data["site"][query],
                    data["device"][query],
                ),
                stage_metrics=[
                    metrics(
                        stages[0, :, j],
                        data["target"][query],
                        data["patient"][query],
                        data["site"][query],
                    )
                    for j in range(stages.shape[2])
                ],
                transforms=transforms,
                doses=doses,
            )
            metadata.append(record)
    assert len(metadata) == 63
    result = dict(
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        records=metadata,
        evidence="Repeated original TRAIN roles; camera and people confounded; no ordinary-phone end-to-end validation",
    )
    path = RUN / "results.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "all"), default="all")
    args = parser.parse_args()
    setup("cuda")
    data = load_data()
    assert data["color"].shape == (966, 36) and data["target"].shape == (966, 3)
    if args.stage == "preflight":
        assert not (RUN / "source_lock.json").exists(), "use normal resume for frozen runs"
        rr = references()
        assert len(rr) == 18
        print(
            dict(
                stage="preflight",
                data_sha256=sha(CACHE),
                families=SPECS,
                inner_banks=45,
                inner_trajectories=270,
                inner_models=810,
                choices=15,
                final_fits=45,
                exact_references=len(rr),
                gpu=torch.cuda.get_device_name(),
            ),
            flush=True,
        )
        return
    start = time.perf_counter()
    source = freeze()
    try:
        inner(data, source)
        selection = select(data, source)
        final(data, source, selection)
        evaluate(data, source, selection)
    except BaseException as error:
        write_json(RUN / "progress.json", dict(status="failed", pid=os.getpid(), error=repr(error)))
        raise
    write_json(
        RUN / "progress.json",
        dict(
            status="complete",
            pid=os.getpid(),
            seconds=time.perf_counter() - start,
            results_sha256=sha(RUN / "results.json"),
            source_lock_sha256=source,
        ),
    )
    print("ND COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
