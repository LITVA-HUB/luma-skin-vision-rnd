"""TRAIN-only architectural screen with selection preceding all held-role access."""

from __future__ import annotations

import argparse
import os
import shutil
import time

import numpy as np
import torch
from chromaseed_architecture_scale import (
    BATCH,
    HORIZON,
    RATES,
    SEEDS,
    SLOTS,
    VARIANTS,
    capacity,
    fit,
    predict_torch,
    specs,
)
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context
from chromaseed_patch8_numpy import choose
from chromaseed_refine_train import setup
from chromaseed_widen_early_run import OUT as WE_OUT
from chromaseed_widen_early_run import RUN as WE
from chromaseed_widen_run import check_map, load_data
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json

RUN = ROOT / "experiments/runs/chromaseed_architecture_scale_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1"
TIMES = (128, 512, 2048)


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = nz(path)
        assert set(old) == set(payload)
        for k in old:
            np.testing.assert_array_equal(old[k], payload[k])
        return
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as f:
        np.savez(f, **payload)
    temp.replace(path)


def bank_path(role, variant, fold=None):
    return (
        RUN
        / ("final" if fold is None else "inner")
        / role
        / variant
        / ("bank" if fold is None else f"fold{fold}")
    )


def freeze():
    parent = js(WE_OUT / "verification.json")
    assert parent["passed"]
    sources = {**parent["sources"], **parent["postprocess_sources"]}
    inputs = {**parent["inputs"], **parent["artifact_sha256"]}
    inputs[(WE_OUT / "verification.json").relative_to(ROOT).as_posix()] = sha(
        WE_OUT / "verification.json"
    )
    for name in (
        "preflight_fp32_source.py",
        "preflight_fp32.json",
        "preflight_bf16_source.py",
        "preflight_bf16.json",
        "preflight.json",
    ):
        p = RUN / name
        inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    for p in (
        "scripts/chromaseed_architecture_scale.py",
        "scripts/chromaseed_architecture_scale_run.py",
        "tests/test_chromaseed_architecture_scale.py",
        "docs/research/chromaseed_architecture_scale_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    check_map({**sources, **inputs})
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    preflight = js(RUN / "preflight.json")
    assert preflight["source_sha256"] == sha(ROOT / "scripts/chromaseed_architecture_scale.py")
    assert preflight["passed"]
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        variants={
            v: dict(specs=[list(s) for s in specs(v)], parameters=capacity(v)) for v in VARIANTS
        },
        slots=[list(s) for s in SLOTS],
        horizon=HORIZON,
        checkpoints=list(TIMES),
        batch_size=BATCH,
        inner_banks=63,
        final_banks=21,
        candidate_scores=132,
        choices=24,
        preflight_sha256=sha(RUN / "preflight.json"),
        torch=torch.__version__,
        numpy=np.__version__,
        gpu=torch.cuda.get_device_name(),
        evidence="reused TRAIN-only exploratory roles",
    )
    path = RUN / "source_lock.json"
    if path.exists():
        assert js(path) == value, "AS source/config changed after freeze"
    else:
        assert shutil.disk_usage(ROOT).free > 35_000_000_000, (
            "reserve experiment storage before fitting"
        )
        write_json(path, value)
    return sha(path)


def train_bank(data, source, role, variant, fold=None, steps=2048):
    ix, query, warm, parents = context(data, role, fold)
    path = bank_path(role, variant, fold)
    selector = None if fold is not None else sha(RUN / "selections.json")
    checkpoints = TIMES if fold is not None else (steps,)
    if (path / "receipt.json").exists():
        old = js(path / "receipt.json")
        assert (
            old["source_lock_sha256"] == source
            and old["selection_sha256"] == selector
            and old["warm_parent_sha256"] == parents
        )
        for name, digest in old["files"].items():
            assert sha(path / name) == digest
        print("AS REUSE", role, variant, fold, flush=True)
        return
    assert shutil.disk_usage(ROOT).free > 15_000_000_000, "storage reserve reached"
    print("AS START", role, variant, fold, steps, flush=True)

    def progress(item):
        if item["step"] in checkpoints or item["step"] % 512 == 0:
            print(
                "AS TRAIN", role, variant, fold, item["step"], round(item["seconds"], 2), flush=True
            )

    models, info = fit(
        data["color"][ix],
        data["tokens"][ix],
        data["target"][ix],
        weights_for(data["patient"][ix], data["site"][ix]),
        warm,
        variant,
        steps,
        checkpoints,
        "cuda",
        "cuda_graph",
        progress,
    )
    files = {}

    def write(name, payload):
        save(path / name, payload)
        files[name] = sha(path / name)

    write("warm_models.npz", flatten({str(i): m for i, m in enumerate(warm)}))
    for step, ms in models.items():
        write(f"models_{step}.npz", flatten({str(i): m for i, m in enumerate(ms)}))
        if fold is not None:
            pred = np.stack(
                [predict_torch(m, data["color"][query], data["tokens"][query]) for m in ms]
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
    )
    write_json(path / "receipt.json", info)
    print("AS SAVED", role, variant, fold, flush=True)


def oof(role, variant, rate, step):
    parts = [nz(bank_path(role, variant, f) / f"oof_{step}.npz") for f in range(3)]
    rows = np.concatenate([p["row_indices"] for p in parts])
    order = np.argsort(rows)
    pred = np.concatenate([p["predictions"] for p in parts], 1)[:, order]
    return rows[order], pred[[2 * i + RATES.index(rate) for i in range(3)]]


def select(data, source):
    previous = js(WE / "selections.json")
    value = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        candidates = []
        for variant in VARIANTS:
            for rate in RATES:
                for step in TIMES:
                    rows, pred = oof(role, variant, rate, step)
                    np.testing.assert_array_equal(rows, ix)
                    mm = [
                        metrics(p, data["target"][ix], data["patient"][ix], data["site"][ix])
                        for p in pred
                    ]
                    candidates.append(
                        dict(
                            variant=variant,
                            step=step,
                            lr=rate,
                            parameters=capacity(variant),
                            numeric_bytes=4 * capacity(variant) + 458,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
        for variant, parent in (
            ("np", next(c for c in previous["roles"][role]["candidates"] if c["variant"] == "np")),
            ("we", previous["roles"][role]["policies"]["overall"]),
        ):
            candidates.append(dict(parent, variant=variant, parent_variant=parent["variant"]))
        assert len(candidates) == 44
        policies = {v: choose([c for c in candidates if c["variant"] == v]) for v in VARIANTS}
        policies["overall"] = choose(candidates)
        value["roles"][role] = dict(candidates=candidates, policies=policies)
    path = RUN / "selections.json"
    if path.exists():
        assert js(path) == value
    else:
        write_json(path, value)
    print("AS SELECTION FROZEN", sha(path), flush=True)
    return value


def evaluate(data, source, selection):
    previous = js(WE / "results.json")
    records = []
    for role, (_, mask) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(mask)
        for variant in VARIANTS:
            c = selection["roles"][role]["policies"][variant]
            path = bank_path(role, variant) / f"models_{c['step']}.npz"
            bank = nz(path)
            for si, seed in enumerate(SEEDS):
                slot = 2 * si + RATES.index(c["lr"])
                model = unpack(bank, str(slot))
                name = f"{variant}_s{seed}"
                dest = RUN / "models" / role / f"{name}.npz"
                save(dest, model)
                passes = predict_torch(
                    model, data["color"][rows], data["tokens"][rows], all_passes=True
                )
                pred = passes[:, -1]
                out = RUN / "evaluated" / role / f"{name}.npz"
                save(out, dict(row_indices=rows, predictions=pred, passes=passes))
                records.append(
                    dict(
                        role=role,
                        variant=variant,
                        seed=seed,
                        step=c["step"],
                        lr=c["lr"],
                        source_path=path.relative_to(ROOT).as_posix(),
                        source_sha256=sha(path),
                        slot=slot,
                        model=dest.relative_to(ROOT).as_posix(),
                        model_sha256=sha(dest),
                        output=out.relative_to(ROOT).as_posix(),
                        output_sha256=sha(out),
                        parameters=capacity(variant),
                        numeric_bytes=sum(
                            v.nbytes for v in model.values() if v.dtype.kind in "biufc"
                        ),
                        metrics=metrics(
                            pred, data["target"][rows], data["patient"][rows], data["site"][rows]
                        ),
                        overall=selection["roles"][role]["policies"]["overall"]["variant"]
                        == variant,
                    )
                )
            print("AS EVALUATED", role, variant, flush=True)
        for variant in ("np", "we"):
            old = [
                r
                for r in previous["records"]
                if r["role"] == role
                and (r["kind"] == "np" if variant == "np" else "overall" in r["policies"])
            ]
            assert len(old) == 3
            for r in old:
                records.append(
                    dict(
                        role=role,
                        variant=variant,
                        seed=r["seed"],
                        step=r["step"],
                        lr=r["lr"],
                        imported_we_record=r,
                        metrics=r["metrics"],
                        overall=selection["roles"][role]["policies"]["overall"]["variant"]
                        == variant,
                    )
                )
    assert len(records) == 81 and sum(r["overall"] for r in records) == 9
    result = dict(
        source_lock_sha256=source, selection_sha256=sha(RUN / "selections.json"), records=records
    )
    write_json(RUN / "results.json", result)
    return result


def preflight():
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_chromaseed_architecture_scale import fixture

    assert not (RUN / "source_lock.json").exists()
    x, t, y, warm = fixture()
    results = []
    for variant in ("patch5m", "soft5m", "dynamic5m", "pool5m"):
        _, info = fit(x, t, y, np.ones(len(x)), warm, variant, 64, (64,), "cuda", "cuda_graph")
        results.append(info)
        print(
            "AS SYNTHETIC PREFLIGHT",
            variant,
            info["full_bank_seconds"],
            info["cuda_peak_allocated_bytes"],
            flush=True,
        )
    assert max(r["cuda_peak_allocated_bytes"] for r in results) < 6_500_000_000
    write_json(
        RUN / "preflight.json",
        dict(
            passed=True,
            synthetic=True,
            source_sha256=sha(ROOT / "scripts/chromaseed_architecture_scale.py"),
            records=results,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    setup("cuda")
    if args.preflight:
        preflight()
        return
    assert not (OUT / "verification.json").exists(), "sealed AS writers cannot rerun"
    source = freeze()
    data = load_data()
    start = time.perf_counter()
    write_json(RUN / "job.json", dict(status="running", pid=os.getpid(), source_lock_sha256=source))
    for role in roles(data["patient"], data["device"]):
        for variant in VARIANTS:
            for fold in range(3):
                train_bank(data, source, role, variant, fold)
    selection = select(data, source)
    for role in selection["roles"]:
        for variant in VARIANTS:
            train_bank(
                data,
                source,
                role,
                variant,
                steps=selection["roles"][role]["policies"][variant]["step"],
            )
    evaluate(data, source, selection)
    write_json(
        RUN / "job.json",
        dict(
            status="complete",
            pid=os.getpid(),
            source_lock_sha256=source,
            results_sha256=sha(RUN / "results.json"),
            seconds=time.perf_counter() - start,
        ),
    )
    print("AS COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
