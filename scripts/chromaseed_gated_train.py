"""Frozen banks and person-held-out selection for compact conditional residuals."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from chromaseed_fast_kernel import get_model
from chromaseed_fast_kernel_train import average_metrics, js, nz, row_hash, valid_bank
from chromaseed_gated import (
    FAMILIES,
    SEEDS,
    candidates,
    fit_bank,
    flatten,
    model_id,
    predict,
    unpack,
)
from chromaseed_kernel_train import atomic_npz
from skin_local_search_train import CACHE_HASH, folds_for, metrics, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def lock_sources(run, cache, parent):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("original TRAIN required")
    d = ROOT / "experiments/runs/chromaseed_camera_support_v1/source_lock.json"
    dv = ROOT / "docs/benchmarks/chromaseed_camera_support_v1/verification.json"
    inherited = js(d)
    sources = dict(inherited["sources"])
    for rel, h in sources.items():
        assert sha(ROOT / rel) == h, rel
    for rel in (
        "scripts/chromaseed_gated.py",
        "scripts/chromaseed_gated_train.py",
        "tests/test_chromaseed_gated.py",
        "docs/research/chromaseed_gated_v1_protocol.md",
    ):
        sources[rel] = sha(ROOT / rel)
    paths = [parent / "source_lock.json", parent / "selections.json", parent / "results.json"]
    for directory in sorted((parent / "inner").glob("*/fold*")) + sorted(
        (parent / "final").glob("*/bank")
    ):
        paths.extend([directory / "receipt.json", directory / "models.npz"])
    assert js(dv)["passed"]
    lock = dict(
        sources=sources,
        cache_sha256=CACHE_HASH,
        parent_bindings={p.relative_to(parent).as_posix(): sha(p) for p in paths},
        input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in (d, dv)},
        families=list(FAMILIES),
        seeds=list(SEEDS),
        rank=128,
        base_alpha=0.1,
        width_factor=1.0,
        gate_alpha=0.1,
        residual_lambdas=[0.1, 1.0, 10.0],
        rhos=[0.0, 0.25, 0.5, 1.0],
        numpy=np.__version__,
        scipy=scipy.__version__,
        python=platform.python_version(),
        threads=1,
    )
    path = run / "source_lock.json"
    if path.exists() and js(path) != lock:
        raise ValueError("G lock changed")
    if not path.exists():
        write_json(path, lock)
    return sha(path)


def check_controls(models, parent, rows):
    receipt = js(parent / "receipt.json")
    assert receipt["fit_rows_sha256"] == row_hash(rows)
    assert sha(parent / "models.npz") == receipt["files"]["models.npz"]
    arrays = nz(parent / "models.npz")
    count = 0
    for base, old, step in [("norm", "norm_mse", 0), ("perceptual", "constant_de2", 1)]:
        for seed in SEEDS:
            expected = get_model(arrays, f"{old}_s{seed}_w1_a0_t{step}")
            actual = models[model_id(f"{base}_base", seed)]
            assert set(actual) == set(expected)
            for field in expected:
                np.testing.assert_array_equal(actual[field], expected[field])
            count += 1
    return count


def run_bank(run, data, parent, role, fit, queries, fold, lock):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    directory = run / stage / role / sub
    if valid_bank(directory, lock):
        print(f"REUSE {stage}/{role}/{sub}", flush=True)
        return
    if queries is not None:
        assert not set(data["patient"][fit]) & set(data["patient"][queries])
    started = time.perf_counter()
    write_json(
        run / "progress.json",
        dict(
            status="fitting",
            stage=stage,
            role=role,
            fold=fold,
            pid=os.getpid(),
            updated_unix=time.time(),
        ),
    )
    models, receipt = fit_bank(
        *(data[k][fit] for k in ("color", "target", "patient", "site", "device"))
    )
    receipt["exact_P_base_payloads"] = check_controls(models, parent / stage / role / sub, fit)
    fallback = 0
    for name, record in receipt["models"].items():
        if record["fallback"] == "single_camera":
            base = record["family"].split("_")[0] + "_base"
            expected = models[model_id(base, record["seed"])]
            assert set(models[name]) == set(expected)
            for k in expected:
                np.testing.assert_array_equal(models[name][k], expected[k])
            fallback += 1
    receipt["exact_fallback_payloads"] = fallback
    atomic_npz(directory / "models.npz", flatten(models))
    files = {"models.npz": sha(directory / "models.npz")}
    if queries is not None:
        saved = {"row_indices": queries}
        for name, model in models.items():
            saved["pred__" + name] = predict(model, data["color"][queries])
        atomic_npz(directory / "oof.npz", saved)
        files["oof.npz"] = sha(directory / "oof.npz")
    receipt.update(
        source_lock_sha256=lock,
        role=role,
        fold=fold,
        fit_rows_sha256=row_hash(fit),
        query_rows_sha256=row_hash(queries) if queries is not None else None,
        n_fit_people=len(np.unique(data["patient"][fit])),
        files=files,
        bank_wall_seconds=time.perf_counter() - started,
    )
    write_json(directory / "receipt.json", receipt)
    print(
        f"FIT {stage}/{role}/{sub}:168 stored,{receipt['residual_solutions']} residual solutions,{fallback} exact fallbacks",
        flush=True,
    )


def fit_stage(run, data, parent, lock, final=False):
    if final:
        assert js(run / "selections.json")["source_lock_sha256"] == lock
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        if final:
            run_bank(run, data, parent, role, rows, None, None, lock)
        else:
            folds = folds_for(data["patient"][fit], data["device"][fit])
            for fold in range(3):
                run_bank(
                    run, data, parent, role, rows[folds != fold], rows[folds == fold], fold, lock
                )
    write_json(
        run / ("final_complete.json" if final else "inner_complete.json"),
        dict(
            source_lock_sha256=lock,
            banks=3 if final else 9,
            readouts=504 if final else 1512,
            selection_sha256=sha(run / "selections.json") if final else None,
        ),
    )


def choose(candidates):
    return min(candidates, key=lambda c: (c["person_mean"], c["rho"], -c["residual_lambda"]))


def select_stage(run, data, lock):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == lock
    result = {"source_lock_sha256": lock, "roles": {}}
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        banks = []
        for fold in range(3):
            path = run / "inner" / role / f"fold{fold}"
            assert valid_bank(path, lock)
            banks.append(nz(path / "oof.npz"))
        rows = np.concatenate([b["row_indices"] for b in banks])
        order = np.argsort(rows)
        rows = rows[order]
        np.testing.assert_array_equal(rows, np.flatnonzero(fit))
        choices = {}
        for family in FAMILIES:
            options = []
            for config in candidates(family):
                pred = [
                    np.concatenate([b["pred__" + model_id(family, s, **config)] for b in banks])[
                        order
                    ]
                    for s in SEEDS
                ]
                options.append(dict(family=family, **config, **average_metrics(pred, data, rows)))
            choices[family] = {"selected": choose(options), "candidates": options}
        result["roles"][role] = choices
    path = run / "selections.json"
    if path.exists() and js(path) != result:
        raise ValueError("frozen selection differs")
    write_json(path, result)
    print(f"LOCKED24 family choices {sha(path)}", flush=True)


def evaluate_stage(run, data, lock):
    marker = js(run / "final_complete.json")
    assert marker["source_lock_sha256"] == lock and marker["selection_sha256"] == sha(
        run / "selections.json"
    )
    partitions = roles(data["patient"], data["device"])
    for role in partitions:
        assert valid_bank(run / "final" / role / "bank", lock)
    selected = js(run / "selections.json")
    records = []
    for role, (_, held) in partitions.items():
        rows = np.flatnonzero(held)
        bank = nz(run / "final" / role / "bank/models.npz")
        for family, choice in selected["roles"][role].items():
            c = choice["selected"]
            for seed in SEEDS:
                model = unpack(bank, model_id(family, seed, c["residual_lambda"], c["rho"]))
                assert model
                prediction = predict(model, data["color"][rows])
                name = f"{family}_s{seed}.npz"
                mp, pp = run / "selected" / role / name, run / "evaluated" / role / name
                atomic_npz(mp, model)
                atomic_npz(pp, dict(row_indices=rows, prediction=prediction))
                records.append(
                    dict(
                        role=role,
                        family=family,
                        seed=seed,
                        residual_lambda=c["residual_lambda"],
                        rho=c["rho"],
                        active_gate="correction" in model,
                        numeric_bytes=sum(v.nbytes for v in model.values()),
                        archive_bytes=mp.stat().st_size,
                        model_sha256=sha(mp),
                        prediction_sha256=sha(pp),
                        metrics=metrics(
                            prediction,
                            data["target"][rows],
                            data["patient"][rows],
                            data["site"][rows],
                        ),
                    )
                )
        print(f"EVALUATED {role}:24 models", flush=True)
    assert len(records) == 72
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=lock,
            selection_sha256=sha(run / "selections.json"),
            records=records,
            evidence="historically reused overlapping TRAIN roles",
        ),
    )
    write_json(
        run / "progress.json",
        dict(
            status="fit_evaluation_complete_audit_pending",
            pid=os.getpid(),
            updated_unix=time.time(),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "inner", "select", "final", "evaluate"])
    for option in ("run", "cache"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument(
        "--parent", type=Path, default=ROOT / "experiments/runs/chromaseed_perceptual_v1"
    )
    args = parser.parse_args()
    started = time.perf_counter()
    lock = lock_sources(args.run, args.cache, args.parent)
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    if args.stage in ("run", "inner"):
        fit_stage(args.run, data, args.parent, lock)
    if args.stage in ("run", "select"):
        select_stage(args.run, data, lock)
    if args.stage in ("run", "final"):
        fit_stage(args.run, data, args.parent, lock, True)
    if args.stage in ("run", "evaluate"):
        evaluate_stage(args.run, data, lock)
    if args.stage == "run":
        write_json(
            args.run / "workflow.json",
            dict(
                source_lock_sha256=lock,
                wall_seconds=time.perf_counter() - started,
                scope="includes read/fit/selection/persist/evaluate, excludes imports/audit/runtime; resumed banks reported in command log",
            ),
        )


if __name__ == "__main__":
    main()
