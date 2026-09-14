"""One registered stronger-alpha study on exact NR representations."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian_audit import direct
from chromaseed_gaussian_train import infer, score
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_neural_readout import (
    BASES,
    FAMILIES,
    GROUPS,
    SEEDS,
    choose,
    choose_policy,
    fit_head,
    name_for,
)
from chromaseed_neural_readout_reference import refit
from chromaseed_perceptual_audit import balanced
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
NR = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
NG = ROOT / "experiments/runs/chromaseed_neural_geometry_v1"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
NR_HASH = "3e4079df5d688136ef59be6b599633b0bc7c44d9d21f410eaeefc2aef0699c65"
NG_HASH = "57b1d6b00b51e9f761ad34b2062f28a1186cd403f0ba40a52e24b0d105014e25"


def freeze(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    sources = dict(js(NG / "source_lock.json")["sources"])
    inputs = {}
    for kind, expected in (("neural_readout", NR_HASH), ("neural_geometry", NG_HASH)):
        path = ROOT / f"docs/benchmarks/chromaseed_{kind}_v1/verification.json"
        assert sha(path) == expected and js(path)["passed"]
        inputs[path.relative_to(ROOT).as_posix()] = expected
        receipt = js(path)
        for p, h in {
            **receipt.get("postprocess_sources", {}),
            **receipt["artifact_sha256"],
        }.items():
            assert sha(ROOT / p) == h, p
            inputs[p] = h
    for path in (Path(__file__), ROOT / "docs/research/chromaseed_neural_shrinkage_v1_protocol.md"):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for inherited in (js(NR / "source_lock.json"), js(NG / "source_lock.json")):
        for p, h in {**inherited["sources"], **inherited["input_sha256"]}.items():
            assert sha(ROOT / p) == h, p
    paths = [NR / f for f in ("source_lock.json", "selections.json", "results.json")]
    paths += [NG / f for f in ("source_lock.json", "results.json")]
    paths += [ROOT / "scripts/chromaseed_neural_readout_runtime.py"]
    for path in sorted((NR / "inner").glob("*/fold*")) + sorted((NR / "final").glob("*/bank")):
        paths += [path / "receipt.json"]
        for name, digest in js(path / "receipt.json")["files"].items():
            assert sha(path / name) == digest
            paths.append(path / name)
    inputs.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        alphas=list(ALPHAS),
        expected_bases=504,
        expected_head_refits=5040,
        expected_records=5556,
        expected_exact_NR_controls=3540,
        previous_turn_classification="progress",
        representation_fitting="cached exact NR bases; full construction separately timed",
    )
    lock = run / "source_lock.json"
    if lock.exists():
        assert js(lock) == value, "frozen NS sources or inputs changed"
    else:
        write_json(lock, value)
    return sha(lock)


def bank(run, data, role, fit, query, fold, source):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    path, oldpath = run / stage / role / sub, NR / stage / role / sub
    selection = sha(run / "selections.json") if fold is None else None
    if valid_bank(path, source):
        rec = js(path / "receipt.json")
        assert rec["selection_sha256"] == selection and rec["fit_rows_sha256"] == row_hash(fit)
        assert rec["query_rows_sha256"] == (None if query is None else row_hash(query))
        print(f"REUSE {stage}/{role}/{sub}", flush=True)
        return
    started = time.perf_counter()
    old, bases, originals = (
        js(oldpath / "receipt.json"),
        nz(oldpath / "bases.npz"),
        nz(oldpath / "models.npz"),
    )
    assert old["fit_rows_sha256"] == row_hash(fit)
    if query is not None:
        assert old["query_rows_sha256"] == row_hash(query)
        assert not set(data["patient"][fit]) & set(data["patient"][query])
    w = weights_for(data["patient"][fit], data["site"][fit])
    rw = balanced(data["patient"][fit], data["site"][fit])
    np.testing.assert_allclose(w, rw, rtol=1e-12, atol=1e-12)
    x, y = data["color"][fit], data["target"][fit]
    models, qr_models, metadata = {}, {}, {}
    exact, max_fit, max_query = 0, 0.0, 0.0
    write_json(
        run / "progress.json",
        dict(status="fitting", role=role, stage=stage, fold=fold, pid=os.getpid()),
    )
    for base_name, base_meta in old["bases"].items():
        basis, group, seed = (base_meta[k] for k in ("basis", "group", "seed"))
        base = unpack(bases, base_name)
        for family in FAMILIES:
            for ai, alpha in enumerate(ALPHAS):
                name = name_for(family, basis, group, seed, ai)
                model, diagnostic = fit_head(base, x, y, w, family, alpha)
                reference, qr_info = refit(base, x, y, rw, family, alpha)
                df = float(np.max(abs(infer(model, x) - direct(reference, x))))
                dq = (
                    0.0
                    if query is None
                    else float(
                        np.max(
                            abs(
                                infer(model, data["color"][query])
                                - direct(reference, data["color"][query])
                            )
                        )
                    )
                )
                assert max(df, dq) <= 0.001, (name, df, dq)
                max_fit, max_query = max(max_fit, df), max(max_query, dq)
                if ai < 3:
                    previous = unpack(originals, name)
                    assert set(model) == set(previous)
                    for key in model:
                        np.testing.assert_array_equal(model[key], previous[key])
                    exact += 1
                models[name], qr_models[name] = model, reference
                metadata[name] = dict(
                    family=family,
                    basis=basis,
                    group=group,
                    seed=seed,
                    alpha=alpha,
                    alpha_index=ai,
                    base_name=base_name,
                    origin="fit_head_on_exact_NR_basis",
                    fit_diagnostics=diagnostic,
                    qr_info=qr_info,
                    qr_fit_max_lab=df,
                    qr_query_max_lab=dq,
                    numeric_bytes=sum(v.nbytes for v in model.values()),
                )
    for name, meta in old["models"].items():
        if meta["family"] not in FAMILIES:
            models[name] = unpack(originals, name)
            metadata[name] = dict(meta, origin="exact_NR_reference")
            exact += 1
    assert (len(models), len(qr_models), exact) == (463, 420, 295)
    atomic_npz(path / "models.npz", flatten(models))
    atomic_npz(path / "qr_models.npz", flatten(qr_models))
    files = {f: sha(path / f) for f in ("models.npz", "qr_models.npz")}
    if query is not None:
        oof = dict(row_indices=query)
        previous = nz(oldpath / "oof.npz")
        for name, model in models.items():
            oof["pred__" + name] = infer(model, data["color"][query])
            if "pred__" + name in previous:
                np.testing.assert_allclose(
                    oof["pred__" + name], previous["pred__" + name], rtol=0, atol=1e-9
                )
        atomic_npz(path / "oof.npz", oof)
        files["oof.npz"] = sha(path / "oof.npz")
    write_json(
        path / "receipt.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=selection,
            role=role,
            fold=fold,
            fit_rows_sha256=row_hash(fit),
            query_rows_sha256=None if query is None else row_hash(query),
            n_fit_rows=len(fit),
            models=metadata,
            exact_NR_models=exact,
            qr_head_refits=420,
            qr_fit_max_lab=max_fit,
            qr_query_max_lab=max_query,
            files=files,
            bank_wall_seconds=time.perf_counter() - started,
        ),
    )
    print(f"FITTED {stage}/{role}/{sub}:420 heads/QR refits;295 exact NR models", flush=True)


def fit_stage(run, data, source, final=False):
    if final:
        assert js(run / "selections.json")["source_lock_sha256"] == source
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        if final:
            bank(run, data, role, rows, None, None, source)
        else:
            folds = folds_for(data["patient"][rows], data["device"][rows])
            for fold in range(3):
                bank(run, data, role, rows[folds != fold], rows[folds == fold], fold, source)
    write_json(
        run / ("final_complete.json" if final else "inner_complete.json"),
        dict(
            source_lock_sha256=source,
            banks=3 if final else 9,
            records=(3 if final else 9) * 463,
            selection_sha256=sha(run / "selections.json") if final else None,
        ),
    )


def select(run, data, source):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == source
    old = js(NR / "selections.json")
    result = dict(source_lock_sha256=source, roles={}, references={})
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows, merged, seen = np.flatnonzero(fit), {}, []
        for fold in range(3):
            path = run / "inner" / role / f"fold{fold}"
            oof, rec = nz(path / "oof.npz"), js(path / "receipt.json")
            idx = np.searchsorted(rows, oof["row_indices"])
            np.testing.assert_array_equal(rows[idx], oof["row_indices"])
            seen.extend(idx.tolist())
            for name in rec["models"]:
                if name not in merged:
                    merged[name] = np.empty((len(rows), 3))
                merged[name][idx] = oof["pred__" + name]
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
                        item = dict(
                            group=group,
                            family=family,
                            basis=basis,
                            alpha=alpha,
                            alpha_index=ai,
                            clean=float(np.mean([s["clean"] for s in scores])),
                            p90=float(np.mean([s["p90"] for s in scores])),
                            seed_scores=scores,
                        )
                        if ai < 3:
                            original = old["roles"][role][group][family]["bases"][basis][
                                "candidates"
                            ][ai]
                            assert max(abs(item[k] - original[k]) for k in ("clean", "p90")) <= 2e-8
                        candidates.append(item)
                    choices[basis] = dict(candidates=candidates, selected=choose(candidates))
                result["roles"][role][group][family] = dict(
                    bases=choices, policy=choose_policy([c["selected"] for c in choices.values()])
                )
        result["references"][role] = {
            name: score(pred, data["target"][rows], data["patient"][rows])
            for name, pred in merged.items()
            if name.startswith(("unchanged", "fg_")) or name == "constant"
        }
        for name, scores in result["references"][role].items():
            assert (
                max(abs(scores[k] - old["references"][role][name][k]) for k in ("clean", "p90"))
                <= 2e-8
            )
    path = run / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    print("FROZEN84 alpha choices/12 policies from420 scores;252 old scores reproduced", flush=True)


def evaluate(run, data, source):
    selection = js(run / "selections.json")
    selection_hash = sha(run / "selections.json")
    assert js(run / "final_complete.json")["selection_sha256"] == selection_hash
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        path = run / "final" / role / "bank"
        arrays, receipt = nz(path / "models.npz"), js(path / "receipt.json")
        transformed = [affine_features(data["color"][rows], s["dose"], s["anchor"]) for s in grid()]
        for name, meta in receipt["models"].items():
            head = meta["family"] in FAMILIES
            entry = selection["roles"][role][meta["group"]][meta["family"]] if head else None
            if (
                head
                and meta["alpha_index"] != entry["bases"][meta["basis"]]["selected"]["alpha_index"]
            ):
                continue
            model = unpack(arrays, name)
            output = np.stack([infer(model, x) for x in transformed])
            tm, doses = summaries(
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
                    policy_selected=head and meta["basis"] == entry["policy"]["basis"],
                    archive_bytes=mp.stat().st_size,
                    model_sha256=sha(mp),
                    prediction_sha256=sha(pp),
                    metrics=metrics(
                        output[0], data["target"][rows], data["patient"][rows], data["site"][rows]
                    ),
                    transforms=tm,
                    doses=doses,
                )
            )
        print(f"EVALUATED {role}:127 cases x33 transforms", flush=True)
    assert len(records) == 381
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=selection_hash,
            records=records,
            evidence="Reused original TRAIN only; cached-basis stronger-penalty study; full construction separately timed",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "inner", "select", "final", "evaluate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    source = freeze(args.run, args.cache)
    assert not (
        ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1/verification.json"
    ).exists(), "sealed study is read-only"
    data = nz(args.cache)
    if args.stage in ("run", "inner"):
        fit_stage(args.run, data, source)
    if args.stage in ("run", "select"):
        select(args.run, data, source)
    if args.stage in ("run", "final"):
        fit_stage(args.run, data, source, True)
    if args.stage in ("run", "evaluate"):
        evaluate(args.run, data, source)
    write_json(
        args.run / "workflow.json",
        dict(
            source_lock_sha256=source,
            wall_seconds=time.perf_counter() - started,
            pid=os.getpid(),
            exit_code=0,
            scope="Cached NR bases; actual head fits and independent QR checks, selection/evaluation/I/O; excludes hidden training, imports and dedicated full-cost runtime",
        ),
    )
    write_json(
        args.run / "progress.json",
        dict(status="fit_evaluation_complete_audit_pending", pid=os.getpid()),
    )


if __name__ == "__main__":
    main()
