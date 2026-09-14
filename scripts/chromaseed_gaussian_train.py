"""Freeze TG, train nested traces, choose settings, then fit and evaluate final models."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_feature_groups import predict as fg_predict
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian import (
    CHECKPOINTS,
    GROUPS,
    METHODS,
    PARAMETERS,
    SEEDS,
    choose,
    model_id,
    predict,
    train_block,
)
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
FG = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
FG_VERIFICATION = "459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908"


def lock_sources(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    vp = ROOT / "docs/benchmarks/chromaseed_feature_groups_v1/verification.json"
    assert sha(vp) == FG_VERIFICATION and js(vp)["passed"]
    inherited = js(FG / "source_lock.json")
    for p, h in {
        **inherited["sources"],
        **inherited["input_sha256"],
        **js(vp)["postprocess_sources"],
    }.items():
        assert sha(ROOT / p) == h, p
    sources = dict(inherited["sources"])
    for p in (
        "scripts/chromaseed_gaussian.py",
        "scripts/chromaseed_gaussian_numpy.py",
        "scripts/chromaseed_gaussian_reference.py",
        "scripts/chromaseed_gaussian_train.py",
        "tests/test_chromaseed_gaussian.py",
        "docs/research/chromaseed_gaussian_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    paths = [
        vp,
        *(FG / p for p in ("source_lock.json", "selections.json", "results.json")),
        ROOT / "scripts/chromaseed_feature_groups_verify.py",
        ROOT / "scripts/skin_he_audit_report.py",
    ]
    for p in (
        "chromaseed_feature_groups_next_decision.md",
        "chromaseed_forum_methods_data_2026-09-13.md",
        "chromaseed_forum_candidates_2026-09-13.json",
    ):
        paths.append(ROOT / "docs/research" / p)
    for directory in sorted((FG / "inner").glob("*/fold*")) + sorted((FG / "final").glob("*/bank")):
        paths += [directory / "models.npz", directory / "receipt.json"]
        if (directory / "oof.npz").exists():
            paths.append(directory / "oof.npz")
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    assert len(sources) == 79 and len(inputs) == 42, (len(sources), len(inputs))
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        groups=GROUPS,
        methods=list(METHODS),
        seeds=list(SEEDS),
        parameters={k: list(v) for k, v in PARAMETERS.items()},
        checkpoints=list(CHECKPOINTS),
        hidden_units=64,
        numeric_dtype="float64; exported float32 means",
        variance_floor=1e-12,
        order_seed_offset=600017,
        final_settings=grid(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        threads=1,
        previous_turn_classification="progress",
    )
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == value, "frozen TG source/input change"
    else:
        write_json(path, value)
    return sha(path)


def controls(data, fit, role, stage, sub):
    path = FG / stage / role / sub
    parent = js(path / "receipt.json")
    assert parent["fit_rows_sha256"] == row_hash(fit)
    for p, h in parent["files"].items():
        assert sha(path / p) == h
    old = nz(path / "models.npz")
    models, metadata = {}, {}
    for group in GROUPS:
        for seed in SEEDS:
            source_name = f"norm_static_{group}_s{seed}_a0"
            name = f"fg_norm_static_{group}_s{seed}"
            models[name] = unpack(old, source_name)
            metadata[name] = dict(
                method="fg_norm_static",
                group=group,
                seed=seed,
                parameter=0.1,
                parameter_index=0,
                epoch=None,
                origin="FG",
                source_model_id=source_name,
            )
    models["constant"] = unpack(old, "constant")
    metadata["constant"] = dict(
        method="constant",
        group="constant",
        seed=None,
        parameter=None,
        parameter_index=None,
        epoch=None,
        origin="FG",
        source_model_id="constant",
    )
    return models, metadata


def infer(model, x):
    return predict(model, x) if "w1" in model else fg_predict(model, x)


def run_bank(run, data, role, fit, query, fold, source):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    path = run / stage / role / sub
    selection_hash = sha(run / "selections.json") if fold is None else None
    if valid_bank(path, source):
        rec = js(path / "receipt.json")
        assert rec["selection_sha256"] == selection_hash and rec["fit_rows_sha256"] == row_hash(fit)
        assert rec["query_rows_sha256"] == (None if query is None else row_hash(query))
        print(f"REUSE {stage}/{role}/{sub}", flush=True)
        return
    if query is not None:
        assert not set(data["patient"][fit]) & set(data["patient"][query])
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
    selected = js(run / "selections.json") if fold is None else None
    models, metadata, trajectories = {}, {}, []
    weights = weights_for(data["patient"][fit], data["site"][fit])
    for method in METHODS:
        for group in GROUPS:
            settings = (
                PARAMETERS[method]
                if selected is None
                else (selected["roles"][role][method][group]["selected"]["parameter"],)
            )
            cases = [
                dict(method=method, parameter=parameter, seed=seed)
                for seed in SEEDS
                for parameter in settings
            ]
            block, records = train_block(
                data["color"][fit], data["target"][fit], weights, group, cases
            )
            offset = len(trajectories)
            trajectories.extend(records)
            for (index, epoch), model in block.items():
                case = cases[index]
                name = model_id(method, group, case["seed"], case["parameter"], epoch)
                models[name] = model
                metadata[name] = dict(
                    **case,
                    group=group,
                    epoch=epoch,
                    origin="fit",
                    parameter_index=PARAMETERS[method].index(case["parameter"]),
                    trajectory_index=offset + index,
                )
            print(
                f"FIT {stage}/{role}/{sub} {method}/{group}: {len(cases)} trajectories x64 epochs",
                flush=True,
            )
    old, old_meta = controls(data, fit, role, stage, sub)
    models.update(old)
    metadata.update(old_meta)
    assert len(models) == (223 if fold is not None else 79)
    for name, model in models.items():
        metadata[name]["numeric_bytes"] = sum(v.nbytes for v in model.values())
    atomic_npz(path / "models.npz", flatten(models))
    files = {"models.npz": sha(path / "models.npz")}
    if query is not None:
        saved = {"row_indices": query}
        for name, model in models.items():
            saved["pred__" + name] = infer(model, data["color"][query])
        parent_oof = nz(FG / stage / role / sub / "oof.npz")
        np.testing.assert_array_equal(query, parent_oof["row_indices"])
        for name, meta in old_meta.items():
            np.testing.assert_array_equal(
                saved["pred__" + name], parent_oof["pred__" + meta["source_model_id"]]
            )
        atomic_npz(path / "oof.npz", saved)
        files["oof.npz"] = sha(path / "oof.npz")
    receipt = dict(
        source_lock_sha256=source,
        selection_sha256=selection_hash,
        role=role,
        fold=fold,
        fit_rows_sha256=row_hash(fit),
        query_rows_sha256=None if query is None else row_hash(query),
        n_fit_rows=len(fit),
        n_fit_people=len(np.unique(data["patient"][fit])),
        models=metadata,
        trajectories=trajectories,
        imported_FG_models=7,
        examples_seen_all_trajectories=sum(
            t["checkpoints"][-1]["examples_seen"] for t in trajectories
        ),
        files=files,
        bank_wall_seconds=time.perf_counter() - started,
    )
    write_json(path / "receipt.json", receipt)


def fit_stage(run, data, source, final=False):
    if final:
        assert js(run / "selections.json")["source_lock_sha256"] == source
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        if final:
            run_bank(run, data, role, rows, None, None, source)
        else:
            folds = folds_for(data["patient"][rows], data["device"][rows])
            for fold in range(3):
                run_bank(run, data, role, rows[folds != fold], rows[folds == fold], fold, source)
    write_json(
        run / ("final_complete.json" if final else "inner_complete.json"),
        dict(
            source_lock_sha256=source,
            selection_sha256=sha(run / "selections.json") if final else None,
            banks=3 if final else 9,
            records=237 if final else 2007,
            trajectories=54 if final else 486,
        ),
    )


def score(pred, y, person):
    errors = delta_e00(pred, y)
    return dict(
        clean=float(np.mean([errors[person == p].mean() for p in np.unique(person)])),
        p90=float(np.quantile(errors, 0.9)),
    )


def select_stage(run, data, source):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == source
    result = dict(source_lock_sha256=source, roles={}, references={})
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        merged, seen = {}, []
        for fold in range(3):
            path = run / "inner" / role / f"fold{fold}"
            oof, rec = nz(path / "oof.npz"), js(path / "receipt.json")
            indices = np.searchsorted(rows, oof["row_indices"])
            np.testing.assert_array_equal(rows[indices], oof["row_indices"])
            seen.extend(indices.tolist())
            for name in rec["models"]:
                if name not in merged:
                    merged[name] = np.empty((len(rows), 3))
                merged[name][indices] = oof["pred__" + name]
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))
        result["roles"][role] = {}
        for method in METHODS:
            result["roles"][role][method] = {}
            for group in GROUPS:
                candidates = []
                for pindex, parameter in enumerate(PARAMETERS[method]):
                    for epoch in CHECKPOINTS:
                        scores = [
                            score(
                                merged[model_id(method, group, seed, parameter, epoch)],
                                data["target"][rows],
                                data["patient"][rows],
                            )
                            for seed in SEEDS
                        ]
                        candidates.append(
                            dict(
                                method=method,
                                group=group,
                                parameter=parameter,
                                parameter_index=pindex,
                                epoch=epoch,
                                clean=float(np.mean([s["clean"] for s in scores])),
                                p90=float(np.mean([s["p90"] for s in scores])),
                                seed_scores=scores,
                            )
                        )
                result["roles"][role][method][group] = dict(
                    candidates=candidates, selected=choose(candidates)
                )
        result["references"][role] = {
            name: score(pred, data["target"][rows], data["patient"][rows])
            for name, pred in merged.items()
            if name.startswith("fg_") or name == "constant"
        }
    path = run / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    print("FROZEN18 method/group settings from216 candidate scores", flush=True)


def evaluate_stage(run, data, source):
    selection = js(run / "selections.json")
    selection_hash = sha(run / "selections.json")
    assert js(run / "final_complete.json")["selection_sha256"] == selection_hash
    records, curves = [], []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        path = run / "final" / role / "bank"
        bank, receipt = nz(path / "models.npz"), js(path / "receipt.json")
        transformed = [affine_features(data["color"][rows], s["dose"], s["anchor"]) for s in grid()]
        for name, meta in receipt["models"].items():
            model = unpack(bank, name)
            chosen = (
                meta["origin"] == "FG"
                or meta["epoch"]
                == selection["roles"][role][meta["method"]][meta["group"]]["selected"]["epoch"]
            )
            clean = infer(model, data["color"][rows])
            mp = run / "models" / role / f"{name}.npz"
            cp = run / "curves" / role / f"{name}.npz"
            atomic_npz(mp, model)
            atomic_npz(cp, dict(prediction=clean, row_indices=rows))
            record = dict(
                role=role,
                name=name,
                **meta,
                selected_for_stress=chosen,
                archive_bytes=mp.stat().st_size,
                model_sha256=sha(mp),
                clean_prediction_sha256=sha(cp),
                metrics=metrics(
                    clean, data["target"][rows], data["patient"][rows], data["site"][rows]
                ),
            )
            curves.append(record)
            if chosen:
                pred = np.stack([infer(model, x) for x in transformed])
                np.testing.assert_array_equal(pred[0], clean)
                transform_metrics, doses = summaries(
                    pred, data["target"][rows], data["patient"][rows], data["device"][rows], None
                )
                pp = run / "evaluated" / role / f"{name}.npz"
                atomic_npz(pp, dict(prediction=pred, row_indices=rows))
                records.append(
                    dict(
                        **record,
                        prediction_sha256=sha(pp),
                        transforms=transform_metrics,
                        doses=doses,
                    )
                )
        print(
            f"EVALUATED {role}:79 clean curves and25 selected/reference cases x33 transforms",
            flush=True,
        )
    assert len(curves) == 237 and len(records) == 75
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=selection_hash,
            curve_records=curves,
            records=records,
            evidence="Reused TRAIN only; Gaussian/Adam learning, no ordinary-phone face validation",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "inner", "select", "final", "evaluate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source = lock_sources(args.run, args.cache)
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    if args.stage in ("run", "inner"):
        fit_stage(args.run, data, source)
    if args.stage in ("run", "select"):
        select_stage(args.run, data, source)
    if args.stage in ("run", "final"):
        fit_stage(args.run, data, source, True)
    if args.stage in ("run", "evaluate"):
        evaluate_stage(args.run, data, source)
    if args.stage == "run":
        write_json(
            args.run / "workflow.json",
            dict(
                source_lock_sha256=source,
                wall_seconds=time.perf_counter() - started,
                pid=os.getpid(),
                exit_code=0,
                scope="Read/fit/select/save/evaluate, excluding imports/audit/runtime. REUSE logs disclose reuse.",
            ),
        )


if __name__ == "__main__":
    main()
