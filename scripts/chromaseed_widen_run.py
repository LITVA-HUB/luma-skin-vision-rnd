"""Matched capacity experiment; inner selection is frozen before held-role evaluation."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context, save_npz
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_neural_prefix_run import check_map
from chromaseed_patch8_fit import token_normalizers
from chromaseed_patch8_numpy import choose, transform_tokens
from chromaseed_patch8_run import RUN as P8
from chromaseed_patch8_run import load_data
from chromaseed_refine_train import setup
from chromaseed_widen import HORIZON, RATES, SEEDS, SLOTS, SPECS, Bank, capacity, fit, predict
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json

RUN = ROOT / "experiments/runs/chromaseed_widen_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_widen_v1"
CHECKPOINTS = (0, 512, 2048, 8192)


def bank_path(role, variant, fold=None):
    return (
        RUN
        / ("final" if fold is None else "inner")
        / role
        / variant
        / ("bank" if fold is None else f"fold{fold}")
    )


def freeze():
    prior = js(P8 / "source_lock.json")
    sources = dict(prior["sources"])
    inputs = dict(prior["input_sha256"])
    for p in (
        "scripts/chromaseed_widen.py",
        "scripts/chromaseed_widen_run.py",
        "tests/test_chromaseed_widen.py",
        "docs/research/chromaseed_widen_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    for p in (
        P8 / "source_lock.json",
        P8 / "selections.json",
        P8 / "results.json",
        P8 / "progress.json",
    ):
        inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    check_map({**sources, **inputs})
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        specs={k: list(v) for k, v in SPECS.items()},
        slots=list(SLOTS),
        horizon=HORIZON,
        checkpoints=list(CHECKPOINTS),
        inner_banks=45,
        final_banks=15,
        trajectories=360,
        checkpoint_records=1440,
        candidates=93,
        choices=18,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        numpy=np.__version__,
        p8_status="primary completed; independent audit and runtime incomplete; not accepted",
    )
    p = RUN / "source_lock.json"
    if p.exists():
        assert js(p) == value, "frozen WIDE sources changed"
    else:
        write_json(p, value)
    return sha(p)


def train_one(data, source, role, variant, fold=None):
    ix, query, warm, parents = context(data, role, fold)
    path = bank_path(role, variant, fold)
    selector = None if fold is not None else sha(RUN / "selections.json")
    if (path / "receipt.json").exists():
        old = js(path / "receipt.json")
        assert (
            old["source_lock_sha256"] == source
            and old["selection_sha256"] == selector
            and old["warm_parent_sha256"] == parents
        )
        for n, digest in old["files"].items():
            assert sha(path / n) == digest
        print("WIDE REUSE", role, variant, fold, flush=True)
        return
    print("WIDE START", role, variant, fold, "rows", len(ix), flush=True)

    def progress(item):
        if item["step"] in CHECKPOINTS or item["step"] % 2048 == 0:
            print(
                "WIDE TRAIN",
                role,
                variant,
                fold,
                item["step"],
                round(item["seconds"], 2),
                flush=True,
            )

    models, info = fit(
        data["color"][ix],
        data["tokens"][ix],
        data["target"][ix],
        weights_for(data["patient"][ix], data["site"][ix]),
        warm,
        variant,
        HORIZON,
        CHECKPOINTS,
        "cuda",
        "cuda_graph",
        progress,
    )
    files = {}

    def save(name, payload):
        p = path / name
        save_npz(p, payload)
        files[name] = sha(p)

    save("warm_models.npz", flatten({str(i): m for i, m in enumerate(warm)}))
    for step, ms in models.items():
        save(f"models_{step}.npz", flatten({str(i): m for i, m in enumerate(ms)}))
        if fold is not None:
            output = np.stack([predict(m, data["color"][query], data["tokens"][query]) for m in ms])
            save(f"oof_{step}.npz", dict(row_indices=query, predictions=output))
    save(
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
    print("WIDE SAVED", role, variant, fold, flush=True)


def select(data, source):
    result = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        candidates = []
        baseline = None
        for variant in SPECS:
            values = {}
            for step in CHECKPOINTS:
                parts = [nz(bank_path(role, variant, f) / f"oof_{step}.npz") for f in range(3)]
                rows = np.concatenate([p["row_indices"] for p in parts])
                order = np.argsort(rows)
                np.testing.assert_array_equal(rows[order], ix)
                values[step] = np.concatenate([p["predictions"] for p in parts], axis=1)[:, order]
            if baseline is None:
                baseline = values[0].copy()
            else:
                np.testing.assert_allclose(values[0], baseline, rtol=0, atol=2e-8)
            payload = unpack(nz(bank_path(role, variant, 0) / "models_0.npz"), "0")
            for step in CHECKPOINTS[1:]:
                for li, lr in enumerate(RATES):
                    mm = [
                        metrics(
                            values[step][2 * si + li],
                            data["target"][ix],
                            data["patient"][ix],
                            data["site"][ix],
                        )
                        for si in range(3)
                    ]
                    candidates.append(
                        dict(
                            variant=variant,
                            step=step,
                            lr=lr,
                            **capacity(payload),
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
        mm = [
            metrics(baseline[2 * si], data["target"][ix], data["patient"][ix], data["site"][ix])
            for si in range(3)
        ]
        candidates.append(
            dict(
                variant="np",
                step=0,
                lr=None,
                parameters=643,
                numeric_bytes=2886,
                clean=float(np.mean([m["person_mean"] for m in mm])),
                p90=float(np.mean([m["p90"] for m in mm])),
                seed_metrics=mm,
            )
        )
        assert len(candidates) == 31
        policies = {v: choose([c for c in candidates if c["variant"] == v]) for v in SPECS}
        policies["overall"] = choose(candidates)
        result["roles"][role] = dict(candidates=candidates, policies=policies)
    p = RUN / "selections.json"
    if p.exists():
        assert js(p) == result
    else:
        write_json(p, result)
    print("WIDE SELECTION FROZEN", sha(p), flush=True)
    return result


def evaluate(data, source, selection):
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        query = np.flatnonzero(held)
        x, t, y = data["color"][query], data["tokens"][query], data["target"][query]
        person, site, camera = (data[k][query] for k in ("patient", "site", "device"))

        def save_model(model, name, variant, step, lr, seed, policies, clean=None):
            is_base = variant == "np"
            p, pp = RUN / "models" / role / f"{name}.npz", RUN / "evaluated" / role / f"{name}.npz"
            save_npz(p, model)
            if clean is None:
                clean = base_predict(model, x) if is_base else predict(model, x, t)
            selected = is_base or variant in policies
            output = []
            if selected:
                for s in grid():
                    xx = affine_features(x, s["dose"], s["anchor"])
                    output.append(
                        base_predict(model, xx)
                        if is_base
                        else predict(model, xx, transform_tokens(t, s["dose"], s["anchor"]))
                    )
                output = np.stack(output)
                np.testing.assert_allclose(output[0], clean, rtol=0, atol=2e-8)
            else:
                output = clean[None]
            save_npz(pp, dict(row_indices=query, predictions=output))
            cap = dict(parameters=643, numeric_bytes=2886) if is_base else capacity(model)
            rec = dict(
                role=role,
                name=name,
                variant=variant,
                step=step,
                lr=lr,
                seed=seed,
                policies=policies,
                stress_evaluated=selected,
                baseline_alias=step == 0 and not is_base,
                **cap,
                archive_bytes=p.stat().st_size,
                model_sha256=sha(p),
                prediction_sha256=sha(pp),
                metrics=metrics(clean, y, person, site, camera),
            )
            if selected:
                rec["transforms"], rec["doses"] = summaries(output, y, person, camera, None)
            if not is_base:
                rec["permutation_max_lab"] = float(
                    np.max(abs(predict(model, x, t[:, ::-1].copy()) - clean))
                )
                assert rec["permutation_max_lab"] <= 2e-8
            records.append(rec)

        for variant in SPECS:
            for step in CHECKPOINTS:
                packed = nz(bank_path(role, variant) / f"models_{step}.npz")
                for slot, hp in enumerate(SLOTS):
                    policies = [
                        p
                        for p, c in selection["roles"][role]["policies"].items()
                        if c["variant"] == variant and c["step"] == step and c["lr"] == hp["lr"]
                    ]
                    name = f"{variant}_r{slot % 2}_t{step}_s{hp['seed']}"
                    save_model(
                        unpack(packed, str(slot)),
                        name,
                        variant,
                        step,
                        hp["lr"],
                        hp["seed"],
                        policies,
                    )
                print("WIDE EVALUATED", role, variant, step, flush=True)
        _, _, warm, _ = context(data, role)
        for si, seed in enumerate(SEEDS):
            policies = (
                ["overall"]
                if selection["roles"][role]["policies"]["overall"]["variant"] == "np"
                else []
            )
            save_model(warm[si], f"np_s{seed}", "np", 0, None, seed, policies)
    assert len(records) == 369 and sum(r["stress_evaluated"] for r in records) == 54
    value = dict(
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        records=records,
        evidence="Original TRAIN-only reused roles; instrument-referenced skin-region descriptors, no new phone-face validation",
    )
    p = RUN / "results.json"
    if p.exists():
        assert js(p) == value
    else:
        write_json(p, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "all"), default="all")
    args = parser.parse_args()
    assert not (OUT / "verification.json").exists(), "sealed WIDE; read-only report verifier only"
    setup("cuda")
    data = load_data()
    if args.stage == "preflight":
        assert not (RUN / "source_lock.json").exists()
        maximum = 0.0
        for role in roles(data["patient"], data["device"]):
            for fold in (0, 1, 2, None):
                ix, _, warm, _ = context(data, role, fold)
                prep = token_normalizers(data["tokens"][ix])
                for variant in SPECS:
                    net = Bank(warm, variant)
                    for slot in range(6):
                        output = predict(
                            net.export(slot, warm, prep),
                            data["color"][ix[:3]],
                            data["tokens"][ix[:3]],
                        )
                        maximum = max(
                            maximum,
                            float(
                                np.max(
                                    abs(
                                        output
                                        - base_predict(warm[slot // 2], data["color"][ix[:3]])
                                    )
                                )
                            ),
                        )
        assert maximum <= 2e-8, maximum
        print("WIDE PREFLIGHT360 warm mappings; maxLab", maximum, flush=True)
        return
    start = time.perf_counter()
    source = freeze()
    write_json(RUN / "job.json", dict(status="running", pid=os.getpid(), source_lock_sha256=source))
    for role in roles(data["patient"], data["device"]):
        for variant in SPECS:
            for fold in range(3):
                train_one(data, source, role, variant, fold)
    selection = select(data, source)
    for role in roles(data["patient"], data["device"]):
        for variant in SPECS:
            train_one(data, source, role, variant)
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
    print("WIDE COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
