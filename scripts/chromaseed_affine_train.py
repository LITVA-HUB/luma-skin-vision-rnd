"""Frozen person-disjoint A banks, dual inner policies and all-dose final scoring."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from chromaseed_affine import (
    ALPHAS,
    ETAS,
    FAMILIES,
    G_CONTROLS,
    POLICIES,
    SEEDS,
    choose,
    fit_bank,
    model_id,
    predict,
)
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_gate_stability import affine_features, gate_score, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gated import model_id as g_model_id
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


def selection_grid():
    return [grid()[0]] + [r for r in grid() if r["dose"] == 4 / 255]


def lock_sources(run, cache, parent):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("original TRAIN required")
    gs = ROOT / "experiments/runs/chromaseed_gate_stability_v1/source_lock.json"
    ver = ROOT / "docs/benchmarks/chromaseed_gate_stability_v1/verification.json"
    assert sha(ver) == "2ad0c77feabbe7d63fe5f6ab4ecf905a4705a95014e4cb4dc7c8b703a8c4c738"
    previous = js(ver)
    assert previous["passed"] and sha(gs) == previous["source_lock_sha256"]
    for path, h in {**previous["artifact_sha256"], **previous["postprocess_sources"]}.items():
        assert sha(ROOT / path) == h
    inherited = js(gs)
    sources = dict(inherited["sources"])
    for path, h in {**sources, **inherited["input_sha256"]}.items():
        assert sha(ROOT / path) == h, path
    for path in (
        "scripts/chromaseed_affine.py",
        "scripts/chromaseed_affine_train.py",
        "tests/test_chromaseed_affine.py",
        "docs/research/chromaseed_affine_v1_protocol.md",
    ):
        sources[path] = sha(ROOT / path)
    paths = [
        gs,
        ver,
        parent / "source_lock.json",
        parent / "selections.json",
        parent / "results.json",
    ]
    for directory in sorted((parent / "inner").glob("*/fold*")) + sorted(
        (parent / "final").glob("*/bank")
    ):
        paths += [directory / "models.npz", directory / "receipt.json"]
    for r in js(parent / "results.json")["records"]:
        if r["family"] in G_CONTROLS:
            for part, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
                path = parent / part / r["role"] / f"{r['family']}_s{r['seed']}.npz"
                assert sha(path) == r[field]
                paths.append(path)
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    assert len(sources) == 54 and len(inputs) == 101
    lock = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        families=FAMILIES,
        policies=POLICIES,
        alphas=ALPHAS,
        etas=ETAS,
        seeds=SEEDS,
        g_controls=G_CONTROLS,
        rank=128,
        selection_settings=selection_grid(),
        final_settings=grid(),
        clean_guard=0.05,
        numpy=np.__version__,
        scipy=scipy.__version__,
        python=platform.python_version(),
        threads=1,
    )
    # JSON materialization keeps tuple/list types identical during resumed checks.
    import json

    lock = json.loads(json.dumps(lock))
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == lock, "A source/input lock changed"
    else:
        write_json(path, lock)
    return sha(path)


def references(models, receipt, data, rows, parent, role, stage, sub):
    source = parent / stage / role / sub
    old = js(source / "receipt.json")
    assert old["fit_rows_sha256"] == row_hash(rows)
    assert sha(source / "models.npz") == old["files"]["models.npz"]
    bank = nz(source / "models.npz")
    selected = js(parent / "selections.json")["roles"][role]
    for family in G_CONTROLS:
        choice = selected[family]["selected"]
        for seed in SEEDS:
            prior_id = g_model_id(family, seed, choice["residual_lambda"], choice["rho"])
            name = f"g_{family}_s{seed}"
            models[name] = unpack(bank, prior_id)
            assert models[name]
            receipt["models"][name] = dict(
                family="g_" + family, seed=seed, reference=True, prior_model_id=prior_id
            )
    w = weights_for(data["patient"][rows], data["site"][rows])
    models["constant"] = dict(
        constant_lab=np.average(data["target"][rows], axis=0, weights=w).astype(np.float32)
    )
    receipt["models"]["constant"] = dict(family="constant", seed=None, reference=True)
    receipt.update(imported_G_payloads=12, constant_fits=1)


def run_bank(run, data, parent, role, fit, queries, fold, lock):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    directory = run / stage / role / sub
    if valid_bank(directory, lock):
        print(f"REUSE {stage}/{role}/{sub}", flush=True)
        return
    if queries is not None:
        assert not set(data["patient"][fit]) & set(data["patient"][queries])
    start = time.perf_counter()
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
    references(models, receipt, data, fit, parent, role, stage, sub)
    assert len(models) == 157
    aliases = 0
    for name, r in receipt["models"].items():
        if r.get("fallback"):
            expected = models[
                model_id(
                    r["family"].replace("joint_soft", "static"), r["seed"], r["alpha"], r["eta"]
                )
            ]
            assert set(models[name]) == set(expected)
            for key in expected:
                np.testing.assert_array_equal(models[name][key], expected[key])
            aliases += 1
    atomic_npz(directory / "models.npz", flatten(models))
    files = {"models.npz": sha(directory / "models.npz")}
    if queries is not None:
        xx = [
            affine_features(data["color"][queries], s["dose"], s["anchor"])
            for s in selection_grid()
        ]
        saved = {"row_indices": queries}
        for name, model in models.items():
            saved["pred__" + name] = np.stack([predict(model, x) for x in xx])
        atomic_npz(directory / "oof.npz", saved)
        files["oof.npz"] = sha(directory / "oof.npz")
    receipt.update(
        source_lock_sha256=lock,
        role=role,
        fold=fold,
        exact_joint_static_aliases=aliases,
        fit_rows_sha256=row_hash(fit),
        query_rows_sha256=row_hash(queries) if queries is not None else None,
        n_fit_people=len(np.unique(data["patient"][fit])),
        files=files,
        bank_wall_seconds=time.perf_counter() - start,
    )
    write_json(directory / "receipt.json", receipt)
    print(
        f"FIT {stage}/{role}/{sub}:157 stored,{receipt['new_coefficient_solutions']} solved,{aliases} aliases",
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
            readouts=471 if final else 1413,
            selection_sha256=sha(run / "selections.json") if final else None,
        ),
    )


def score_predictions(prediction, target, person):
    error = delta_e00(prediction, target[None])
    people = np.unique(person)
    worst = error[1:].max(0)
    return dict(
        clean=float(np.mean([error[0][person == p].mean() for p in people])),
        robust=float(np.mean([worst[person == p].mean() for p in people])),
    )


def select_stage(run, data, lock):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == lock
    result = dict(source_lock_sha256=lock, roles={}, references={})
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        inverse = {int(v): i for i, v in enumerate(rows)}
        merged = {}
        seen = []
        for fold in range(3):
            arr = nz(run / "inner" / role / f"fold{fold}/oof.npz")
            index = np.array([inverse[int(v)] for v in arr["row_indices"]])
            seen.extend(index.tolist())
            for key, value in arr.items():
                if key.startswith("pred__"):
                    if key not in merged:
                        merged[key] = np.empty((9, len(rows), 3))
                    merged[key][:, index] = value
        np.testing.assert_array_equal(np.sort(seen), np.arange(len(rows)))

        def scores(name):
            return score_predictions(
                merged["pred__" + name], data["target"][rows], data["patient"][rows]
            )

        result["roles"][role] = {}
        for family in FAMILIES:
            candidates = []
            for eta in ETAS:
                for alpha in ALPHAS:
                    sm = [scores(model_id(family, seed, alpha, eta)) for seed in SEEDS]
                    candidates.append(
                        dict(
                            family=family,
                            alpha=alpha,
                            eta=eta,
                            clean=float(np.mean([r["clean"] for r in sm])),
                            robust=float(np.mean([r["robust"] for r in sm])),
                            seed_scores=sm,
                        )
                    )
            result["roles"][role][family] = dict(
                candidates=candidates,
                policies={policy: choose(candidates, policy) for policy in POLICIES},
                zero_eta_anchor_clean=min(r["clean"] for r in candidates if r["eta"] == 0),
                clean_allowance=0.05,
            )
        rr = {}
        for family in G_CONTROLS:
            sm = [scores(f"g_{family}_s{seed}") for seed in SEEDS]
            rr["g_" + family] = dict(
                clean=float(np.mean([r["clean"] for r in sm])),
                robust=float(np.mean([r["robust"] for r in sm])),
                seed_scores=sm,
            )
        rr["constant"] = scores("constant")
        result["references"][role] = rr
    path = run / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    print("FROZEN24 learned family/policy/role choices;15 fixed reference scores", flush=True)


def evaluation_cases(selected, role):
    cases = []
    for family in FAMILIES:
        for policy in POLICIES:
            chosen = selected["roles"][role][family]["policies"][policy]
            for seed in SEEDS:
                cases.append(
                    dict(
                        family=family,
                        policy=policy,
                        seed=seed,
                        alpha=chosen["alpha"],
                        eta=chosen["eta"],
                        name=f"{family}_{policy}_s{seed}",
                        model_id=model_id(family, seed, chosen["alpha"], chosen["eta"]),
                    )
                )
    for family in G_CONTROLS:
        for seed in SEEDS:
            name = f"g_{family}_s{seed}"
            cases.append(
                dict(family="g_" + family, policy="reference", seed=seed, name=name, model_id=name)
            )
    cases.append(
        dict(family="constant", policy="reference", seed=None, name="constant", model_id="constant")
    )
    assert len(cases) == 37
    return cases


def evaluate_stage(run, data, parent, lock):
    assert js(run / "final_complete.json")["source_lock_sha256"] == lock
    assert js(run / "final_complete.json")["selection_sha256"] == sha(run / "selections.json")
    selected = js(run / "selections.json")
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        bank = nz(run / "final" / role / "bank/models.npz")
        rows = np.flatnonzero(held)
        xx = [affine_features(data["color"][rows], s["dose"], s["anchor"]) for s in grid()]
        for case in evaluation_cases(selected, role):
            model = unpack(bank, case["model_id"])
            prediction = np.stack([predict(model, x) for x in xx])
            score = np.stack([gate_score(model, x) for x in xx]) if "gate_beta" in model else None
            if case["family"].startswith("g_"):
                old_name = f"{case['family'][2:]}_s{case['seed']}.npz"
                original_model = nz(parent / "selected" / role / old_name)
                assert set(original_model) == set(model)
                for key in model:
                    np.testing.assert_array_equal(model[key], original_model[key])
                original = nz(parent / "evaluated" / role / old_name)
                np.testing.assert_array_equal(rows, original["row_indices"])
                np.testing.assert_allclose(prediction[0], original["prediction"], atol=2e-8, rtol=0)
            tt, dd = summaries(
                prediction, data["target"][rows], data["patient"][rows], data["device"][rows], score
            )
            mp, pp = (
                run / "selected" / role / f"{case['name']}.npz",
                run / "evaluated" / role / f"{case['name']}.npz",
            )
            atomic_npz(mp, model)
            atomic_npz(
                pp,
                dict(
                    prediction=prediction,
                    row_indices=rows,
                    **({"scores": score} if score is not None else {}),
                ),
            )
            records.append(
                dict(
                    role=role,
                    **case,
                    numeric_bytes=sum(v.nbytes for v in model.values()),
                    archive_bytes=mp.stat().st_size,
                    active_gate="gate_beta" in model,
                    model_sha256=sha(mp),
                    prediction_sha256=sha(pp),
                    metrics=metrics(
                        prediction[0],
                        data["target"][rows],
                        data["patient"][rows],
                        data["site"][rows],
                    ),
                    transforms=tt,
                    doses=dd,
                )
            )
        print(f"EVALUATED {role}:37 records x33 transforms", flush=True)
    assert len(records) == 111
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=lock,
            selection_sha256=sha(run / "selections.json"),
            records=records,
            evidence="historically reused overlapping TRAIN; synthetic augmentation/robustness is not phone validation",
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
    parser.add_argument("stage", choices=("run", "inner", "select", "final", "evaluate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--parent", type=Path, default=ROOT / "experiments/runs/chromaseed_gated_v1"
    )
    args = parser.parse_args()
    run, parent = args.run.resolve(), args.parent.resolve()
    start = time.perf_counter()
    lock = lock_sources(run, args.cache, parent)
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    if args.stage in ("run", "inner"):
        fit_stage(run, data, parent, lock)
    if args.stage in ("run", "select"):
        select_stage(run, data, lock)
    if args.stage in ("run", "final"):
        fit_stage(run, data, parent, lock, True)
    if args.stage in ("run", "evaluate"):
        evaluate_stage(run, data, parent, lock)
    if args.stage == "run":
        write_json(
            run / "workflow.json",
            dict(
                source_lock_sha256=lock,
                wall_seconds=time.perf_counter() - start,
                scope="read/fit/select/persist/evaluate; excludes interpreter imports/audit/runtime; command log records any resumed banks",
            ),
        )


if __name__ == "__main__":
    main()
