"""Freeze NR, fit matched representation/readout banks, select, then evaluate."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian import model_id
from chromaseed_gaussian_train import infer, score
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_neural_readout import (
    ALPHAS,
    BASES,
    EPOCHS,
    FAMILIES,
    GROUPS,
    PARAMETERS,
    SEEDS,
    basis_spec,
    choose,
    choose_policy,
    fit_head,
    name_for,
    representation_bank,
)
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
TG = ROOT / "experiments/runs/chromaseed_gaussian_v1"
TG_RECEIPT = "19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098"


def lock_sources(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    vp = ROOT / "docs/benchmarks/chromaseed_gaussian_v1/verification.json"
    assert sha(vp) == TG_RECEIPT and js(vp)["passed"]
    inherited = js(TG / "source_lock.json")
    for p, h in {
        **inherited["sources"],
        **inherited["input_sha256"],
        **js(vp)["postprocess_sources"],
    }.items():
        assert sha(ROOT / p) == h, p
    sources = dict(inherited["sources"])
    for p in (
        "scripts/chromaseed_neural_readout.py",
        "scripts/chromaseed_neural_readout_reference.py",
        "scripts/chromaseed_neural_readout_train.py",
        "tests/test_chromaseed_neural_readout.py",
        "docs/research/chromaseed_neural_readout_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    paths = [vp, *(TG / p for p in ("source_lock.json", "selections.json", "results.json"))]
    for p in (
        "scripts/chromaseed_gaussian_verify.py",
        "docs/research/chromaseed_gaussian_next_decision.md",
        "scripts/chromaseed_frozen.py",
        "docs/research/chromaseed_frozen_protocol.md",
        "docs/research/chromaseed_next_decision.md",
        "docs/architecture/chromaseed_model_card.md",
        "docs/research/chromaseed_gaussian_consumer_erratum.md",
    ):
        paths.append(ROOT / p)
    for directory in sorted((TG / "inner").glob("*/fold*")) + sorted((TG / "final").glob("*/bank")):
        paths += [directory / "models.npz", directory / "receipt.json"]
        if (directory / "oof.npz").exists():
            paths.append(directory / "oof.npz")
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    assert (len(sources), len(inputs)) == (84, 44), (len(sources), len(inputs))
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        bases=BASES,
        groups=GROUPS,
        families=FAMILIES,
        seeds=SEEDS,
        alphas=ALPHAS,
        representation_parameters=PARAMETERS,
        epochs=EPOCHS,
        hidden_units=64,
        final_settings=grid(),
        numeric_dtype="FP64 fits; FP32 network payload",
        threads=1,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        previous_turn_classification="progress",
    )
    # Round-trip canonical JSON container types so a saved lock compares identically on reuse.
    import json

    value = json.loads(json.dumps(value))
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == value, "frozen NR source/input change"
    else:
        write_json(path, value)
    return sha(path)


def run_bank(run, data, role, fit, query, fold, source):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    path, parent_path = run / stage / role / sub, TG / stage / role / sub
    selection_hash = sha(run / "selections.json") if fold is None else None
    if valid_bank(path, source):
        rec = js(path / "receipt.json")
        assert rec["selection_sha256"] == selection_hash and rec["fit_rows_sha256"] == row_hash(fit)
        assert rec["query_rows_sha256"] == (None if query is None else row_hash(query))
        print(f"REUSE {stage}/{role}/{sub}", flush=True)
        return
    started = time.perf_counter()
    if query is not None:
        assert not set(data["patient"][fit]) & set(data["patient"][query])
    parent, parent_arrays = js(parent_path / "receipt.json"), nz(parent_path / "models.npz")
    assert parent["fit_rows_sha256"] == row_hash(fit)
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
    weights = weights_for(data["patient"][fit], data["site"][fit])
    models, metadata, bases, basis_metadata, trajectories = {}, {}, {}, {}, []
    for group in GROUPS:
        group_bases, traces = representation_bank(
            data["color"][fit], data["target"][fit], weights, group
        )
        offset = len(trajectories)
        trajectories.extend(traces)
        for (basis, seed), model in group_bases.items():
            method, epoch = basis_spec(basis)
            base_name = f"{basis}_{group}_s{seed}"
            parent_id = (
                None
                if method == "random"
                else model_id(method, group, seed, PARAMETERS[method], epoch)
            )
            available = parent_id in parent["models"]
            if available:
                old = unpack(parent_arrays, parent_id)
                for key in model:
                    np.testing.assert_array_equal(model[key], old[key])
            trace_index = (
                None
                if method == "random"
                else offset
                + next(
                    i for i, t in enumerate(traces) if t["seed"] == seed and t["method"] == method
                )
            )
            bases[base_name] = model
            basis_metadata[base_name] = dict(
                basis=basis,
                group=group,
                seed=seed,
                method=method,
                epoch=epoch,
                trajectory_index=trace_index,
                exact_TG=available,
                parent_model_id=parent_id if available else None,
            )
            if method != "random":
                name = name_for("unchanged", basis, group, seed)
                models[name] = model
                metadata[name] = dict(
                    family="unchanged",
                    basis=basis,
                    group=group,
                    seed=seed,
                    alpha=None,
                    alpha_index=None,
                    base_name=base_name,
                    origin="fit",
                    fit_diagnostics=None,
                )
            for family in FAMILIES:
                for ai, alpha in enumerate(ALPHAS):
                    name = name_for(family, basis, group, seed, ai)
                    fitted, info = fit_head(
                        model, data["color"][fit], data["target"][fit], weights, family, alpha
                    )
                    models[name] = fitted
                    metadata[name] = dict(
                        family=family,
                        basis=basis,
                        group=group,
                        seed=seed,
                        alpha=alpha,
                        alpha_index=ai,
                        base_name=base_name,
                        origin="fit",
                        fit_diagnostics=info,
                    )
        print(f"FIT {stage}/{role}/{sub}/{group}:6 trajectories x16 epochs,126 heads", flush=True)
    for name, meta in parent["models"].items():
        if meta["origin"] != "FG":
            continue
        models[name] = unpack(parent_arrays, name)
        metadata[name] = dict(
            family=meta["method"],
            basis="reference",
            group=meta["group"],
            seed=meta["seed"],
            alpha=meta["parameter"],
            alpha_index=None,
            base_name=None,
            origin="FG",
            parent_model_id=name,
            fit_diagnostics=None,
        )
    assert len(models) == 295 and len(bases) == 42 and len(trajectories) == 12
    for name, model in models.items():
        metadata[name]["numeric_bytes"] = sum(v.nbytes for v in model.values())
    atomic_npz(path / "models.npz", flatten(models))
    atomic_npz(path / "bases.npz", flatten(bases))
    files = {p: sha(path / p) for p in ("models.npz", "bases.npz")}
    if query is not None:
        saved = {"row_indices": query}
        for name, model in models.items():
            saved["pred__" + name] = infer(model, data["color"][query])
        old_oof = nz(parent_path / "oof.npz")
        np.testing.assert_array_equal(query, old_oof["row_indices"])
        for name, meta in metadata.items():
            if meta["origin"] == "FG":
                np.testing.assert_array_equal(
                    saved["pred__" + name], old_oof["pred__" + meta["parent_model_id"]]
                )
            elif meta["family"] == "unchanged":
                bm = basis_metadata[meta["base_name"]]
                if bm["exact_TG"]:
                    np.testing.assert_array_equal(
                        saved["pred__" + name], old_oof["pred__" + bm["parent_model_id"]]
                    )
        atomic_npz(path / "oof.npz", saved)
        files["oof.npz"] = sha(path / "oof.npz")
    write_json(
        path / "receipt.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=selection_hash,
            role=role,
            fold=fold,
            fit_rows_sha256=row_hash(fit),
            query_rows_sha256=None if query is None else row_hash(query),
            n_fit_rows=len(fit),
            n_fit_people=len(np.unique(data["patient"][fit])),
            models=metadata,
            bases=basis_metadata,
            trajectories=trajectories,
            imported_FG_models=7,
            exact_TG_bases=sum(m["exact_TG"] for m in basis_metadata.values()),
            examples_seen_all_trajectories=sum(
                t["checkpoints"][-1]["examples_seen"] for t in trajectories
            ),
            head_solves=252,
            files=files,
            bank_wall_seconds=time.perf_counter() - started,
        ),
    )


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
            records=885 if final else 2655,
            trajectories=36 if final else 108,
            bases=126 if final else 378,
            head_solves=756 if final else 2268,
        ),
    )


def select_stage(run, data, source):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == source
    result = dict(source_lock_sha256=source, roles={}, references={})
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows, merged, seen = np.flatnonzero(fit), {}, []
        for fold in range(3):
            path = run / "inner" / role / f"fold{fold}"
            oof, rec = nz(path / "oof.npz"), js(path / "receipt.json")
            index = np.searchsorted(rows, oof["row_indices"])
            np.testing.assert_array_equal(rows[index], oof["row_indices"])
            seen.extend(index.tolist())
            for name in rec["models"]:
                if name not in merged:
                    merged[name] = np.empty((len(rows), 3))
                merged[name][index] = oof["pred__" + name]
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))
        result["roles"][role] = {}
        for group in GROUPS:
            result["roles"][role][group] = {}
            for family in FAMILIES:
                choices = {}
                for basis in BASES:
                    candidates = []
                    for ai, alpha in enumerate(ALPHAS):
                        scores = [
                            score(
                                merged[name_for(family, basis, group, seed, ai)],
                                data["target"][rows],
                                data["patient"][rows],
                            )
                            for seed in SEEDS
                        ]
                        candidates.append(
                            dict(
                                group=group,
                                family=family,
                                basis=basis,
                                alpha=alpha,
                                alpha_index=ai,
                                clean=float(np.mean([s["clean"] for s in scores])),
                                p90=float(np.mean([s["p90"] for s in scores])),
                                seed_scores=scores,
                            )
                        )
                    choices[basis] = dict(candidates=candidates, selected=choose(candidates))
                result["roles"][role][group][family] = dict(
                    bases=choices, policy=choose_policy([v["selected"] for v in choices.values()])
                )
        result["references"][role] = {
            name: score(pred, data["target"][rows], data["patient"][rows])
            for name, pred in merged.items()
            if name.startswith(("unchanged", "fg_")) or name == "constant"
        }
    path = run / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    print("FROZEN84 alpha choices and12 policies from252 candidates", flush=True)


def evaluate_stage(run, data, source):
    selection, selection_hash = js(run / "selections.json"), sha(run / "selections.json")
    assert js(run / "final_complete.json")["selection_sha256"] == selection_hash
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        path = run / "final" / role / "bank"
        arrays, receipt = nz(path / "models.npz"), js(path / "receipt.json")
        transformed = [affine_features(data["color"][rows], s["dose"], s["anchor"]) for s in grid()]
        for name, meta in receipt["models"].items():
            is_head = meta["family"] in FAMILIES
            entry = selection["roles"][role][meta["group"]][meta["family"]] if is_head else None
            if (
                is_head
                and meta["alpha_index"] != entry["bases"][meta["basis"]]["selected"]["alpha_index"]
            ):
                continue
            model = unpack(arrays, name)
            output = np.stack([infer(model, x) for x in transformed])
            transform_metrics, doses = summaries(
                output, data["target"][rows], data["patient"][rows], data["device"][rows], None
            )
            mp, pp = (
                run / "selected" / role / f"{name}.npz",
                run / "evaluated" / role / f"{name}.npz",
            )
            atomic_npz(mp, model)
            atomic_npz(pp, dict(prediction=output, row_indices=rows))
            records.append(
                dict(
                    role=role,
                    name=name,
                    **meta,
                    policy_selected=is_head and meta["basis"] == entry["policy"]["basis"],
                    archive_bytes=mp.stat().st_size,
                    model_sha256=sha(mp),
                    prediction_sha256=sha(pp),
                    metrics=metrics(
                        output[0], data["target"][rows], data["patient"][rows], data["site"][rows]
                    ),
                    transforms=transform_metrics,
                    doses=doses,
                )
            )
        print(f"EVALUATED {role}:127 selected/reference cases x33 transforms", flush=True)
    assert len(records) == 381
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=selection_hash,
            records=records,
            evidence="Reused TRAIN, analytic output fits on matched compact representations; no ordinary-phone face validation",
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
    write_json(
        args.run / "workflow.json",
        dict(
            source_lock_sha256=source,
            wall_seconds=time.perf_counter() - started,
            pid=os.getpid(),
            exit_code=0,
            scope="Current invocation read/fit/select/save/evaluate, excluding imports/audit/runtime. REUSE logs disclose reuse.",
        ),
    )


if __name__ == "__main__":
    main()
