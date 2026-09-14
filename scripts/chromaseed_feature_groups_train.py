"""Registered TRAIN-only feature groups: inner selection precedes final fits."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from chromaseed_affine import model_id as a_id
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_feature_groups import (
    ALL_GROUPS,
    ALPHAS,
    FAMILIES,
    GROUPS,
    POLICIES,
    SEEDS,
    choose_alpha,
    choose_policy,
    fit_bank,
    gate_score,
    model_id,
    predict,
)
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_projection import model_id as x_id
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
PARENT_A = ROOT / "experiments/runs/chromaseed_affine_v1"
PARENT_X = ROOT / "experiments/runs/chromaseed_projection_v1"
PARENT_C = ROOT / "experiments/runs/chromaseed_crossfit_v1"
C_VERIFICATION = "80c4129565931e204d663d4e833d8513e7d70111b47c965b6ffe63f2eb2c18dd"


def exact(a, b):
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype
        np.testing.assert_array_equal(a[key], b[key])


def lock_sources(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    ver = ROOT / "docs/benchmarks/chromaseed_crossfit_v1/verification.json"
    assert sha(ver) == C_VERIFICATION and js(ver)["passed"]
    inherited = js(PARENT_C / "source_lock.json")
    for p, h in {**inherited["sources"], **inherited["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    sources = dict(inherited["sources"])
    for p in (
        "scripts/chromaseed_feature_groups.py",
        "scripts/chromaseed_feature_groups_numpy.py",
        "scripts/chromaseed_feature_groups_train.py",
        "tests/test_chromaseed_feature_groups.py",
        "docs/research/chromaseed_feature_groups_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    paths = [ver, PARENT_C / "source_lock.json", ROOT / "scripts/chromaseed_crossfit_verify.py"]
    for parent in (PARENT_A, PARENT_X):
        paths += [parent / p for p in ("source_lock.json", "selections.json", "results.json")]
        for directory in sorted((parent / "inner").glob("*/fold*")) + sorted(
            (parent / "final").glob("*/bank")
        ):
            paths += [directory / "models.npz", directory / "receipt.json"]
            if (directory / "oof.npz").exists():
                paths.append(directory / "oof.npz")
    for p in (
        "skin_relational_probe_protocol_v1.md",
        "skin_relational_probe_next_decision.md",
        "skin_support_curve_protocol_v1.md",
        "skin_support_curve_next_decision.md",
        "skin_appearance_inverse_protocol_v1.md",
        "skin_appearance_inverse_next_decision.md",
        "chromaseed_external_models_2026-09-13.md",
    ):
        paths.append(ROOT / "docs/research" / p)
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    assert len(sources) == 73 and len(inputs) == 82, (len(sources), len(inputs))
    value = dict(
        sources=sources,
        input_sha256=inputs,
        cache_sha256=CACHE_HASH,
        groups=GROUPS,
        all_groups=list(ALL_GROUPS),
        families=list(FAMILIES),
        seeds=list(SEEDS),
        alphas=list(ALPHAS),
        policies=list(POLICIES),
        rank=128,
        clean_allowance=0.05,
        p90_allowance=0.1,
        final_settings=grid(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        python=platform.python_version(),
        threads=1,
        previous_turn_classification="progress",
    )
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == value, "frozen feature-group source/input change"
    else:
        write_json(path, value)
    return sha(path)


def parent_bank(parent, role, stage, sub, rows):
    path = parent / stage / role / sub
    rec = js(path / "receipt.json")
    assert rec["fit_rows_sha256"] == row_hash(rows)
    for p, h in rec["files"].items():
        assert sha(path / p) == h
    return nz(path / "models.npz")


def add_controls(models, receipt, data, rows, role, stage, sub):
    aa = parent_bank(PARENT_A, role, stage, sub, rows)
    xx = parent_bank(PARENT_X, role, stage, sub, rows)
    for meta in receipt["models"].values():
        meta["origin"] = "fit"
    raw_count = 0
    for family in FAMILIES:
        for seed in SEEDS:
            for alpha in ALPHAS:
                raw = model_id(family, seed, "raw36", alpha)
                exact(models[raw], unpack(aa, a_id(family, seed, alpha, 0.0)))
                raw_count += 1
                key = model_id(family, seed, "projected16", alpha)
                old_key = x_id(family, seed, "d16_t05", alpha)
                models[key] = unpack(xx, old_key)
                receipt["models"][key] = dict(
                    family=family,
                    seed=seed,
                    group="projected16",
                    alpha=alpha,
                    origin="X",
                    fallback=False,
                    source_model_id=old_key,
                )
    models["constant"] = {
        "constant_lab": np.average(
            data["target"][rows],
            axis=0,
            weights=weights_for(data["patient"][rows], data["site"][rows]),
        ).astype(np.float32)
    }
    receipt["models"]["constant"] = dict(
        family="constant",
        seed=None,
        group="constant",
        alpha=None,
        origin="constant",
        fallback=False,
    )
    receipt.update(exact_A_payloads=raw_count, imported_X_payloads=36, constant_fits=1)


def run_bank(run, data, role, fit, queries, fold, source):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    path = run / stage / role / sub
    if valid_bank(path, source):
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
    add_controls(models, receipt, data, fit, role, stage, sub)
    assert len(models) == 289
    aliases = 0
    for name, meta in receipt["models"].items():
        meta["numeric_bytes"] = sum(v.nbytes for v in models[name].values())
        if meta["fallback"]:
            original = model_id(
                meta["family"].replace("joint_soft", "static"),
                meta["seed"],
                meta["group"],
                meta["alpha"],
            )
            exact(models[name], models[original])
            aliases += 1
    atomic_npz(path / "models.npz", flatten(models))
    files = {"models.npz": sha(path / "models.npz")}
    if queries is not None:
        saved = {"row_indices": queries}
        for name, model in models.items():
            saved["pred__" + name] = predict(model, data["color"][queries])
        aa = nz(PARENT_A / "inner" / role / sub / "oof.npz")
        xx = nz(PARENT_X / "inner" / role / sub / "oof.npz")
        np.testing.assert_array_equal(queries, aa["row_indices"])
        np.testing.assert_array_equal(queries, xx["row_indices"])
        for family in FAMILIES:
            for seed in SEEDS:
                for alpha in ALPHAS:
                    np.testing.assert_allclose(
                        saved["pred__" + model_id(family, seed, "raw36", alpha)],
                        aa["pred__" + a_id(family, seed, alpha, 0)][0],
                        atol=2e-8,
                        rtol=0,
                    )
                    np.testing.assert_array_equal(
                        saved["pred__" + model_id(family, seed, "projected16", alpha)],
                        xx["pred__" + x_id(family, seed, "d16_t05", alpha)],
                    )
        atomic_npz(path / "oof.npz", saved)
        files["oof.npz"] = sha(path / "oof.npz")
    receipt.update(
        source_lock_sha256=source,
        role=role,
        fold=fold,
        exact_joint_static_aliases=aliases,
        fit_rows_sha256=row_hash(fit),
        query_rows_sha256=None if queries is None else row_hash(queries),
        n_fit_people=len(np.unique(data["patient"][fit])),
        files=files,
        bank_wall_seconds=time.perf_counter() - start,
    )
    write_json(path / "receipt.json", receipt)
    print(
        f"FIT {stage}/{role}/{sub}:289 records,{receipt['new_coefficient_solutions']} solves,{aliases} aliases",
        flush=True,
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
            banks=3 if final else 9,
            readouts=867 if final else 2601,
            selection_sha256=sha(run / "selections.json") if final else None,
        ),
    )


def score(pred, target, person):
    errors = delta_e00(pred, target)
    return dict(
        clean=float(np.mean([errors[person == p].mean() for p in np.unique(person)])),
        p90=float(np.quantile(errors, 0.9)),
    )


def select_stage(run, data, source):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == source
    result = dict(source_lock_sha256=source, roles={}, references={})
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        merged, sizes, seen = {}, {}, []
        for fold in range(3):
            path = run / "inner" / role / f"fold{fold}"
            oof, rec = nz(path / "oof.npz"), js(path / "receipt.json")
            index = np.searchsorted(rows, oof["row_indices"])
            np.testing.assert_array_equal(rows[index], oof["row_indices"])
            seen.extend(index.tolist())
            for name, meta in rec["models"].items():
                if name not in merged:
                    merged[name] = np.empty((len(rows), 3))
                merged[name][index] = oof["pred__" + name]
                sizes[name] = max(sizes.get(name, 0), meta["numeric_bytes"])
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))
        result["roles"][role] = {}
        for family in FAMILIES:
            group_rows = {}
            for order, group in enumerate(ALL_GROUPS):
                candidates = []
                for alpha in ALPHAS:
                    names = [model_id(family, s, group, alpha) for s in SEEDS]
                    scores = [
                        score(merged[n], data["target"][rows], data["patient"][rows]) for n in names
                    ]
                    candidates.append(
                        dict(
                            family=family,
                            group=group,
                            order=order,
                            alpha=alpha,
                            clean=float(np.mean([v["clean"] for v in scores])),
                            p90=float(np.mean([v["p90"] for v in scores])),
                            numeric_bytes=max(sizes[n] for n in names),
                            seed_scores=scores,
                        )
                    )
                group_rows[group] = dict(candidates=candidates, selected=choose_alpha(candidates))
            choices = [v["selected"] for v in group_rows.values()]
            result["roles"][role][family] = dict(
                groups=group_rows, policies={p: choose_policy(choices, p) for p in POLICIES}
            )
        result["references"][role] = score(
            merged["constant"], data["target"][rows], data["patient"][rows]
        )
    path = run / "selections.json"
    if path.exists():
        assert js(path) == result
    else:
        write_json(path, result)
    print("FROZEN96 group settings and24 policies from288 candidate scores", flush=True)


def evaluation_cases(selected, role):
    cases = []
    for family in FAMILIES:
        for group in ALL_GROUPS:
            chosen = selected["roles"][role][family]["groups"][group]["selected"]
            for seed in SEEDS:
                name = f"{family}_{group}_s{seed}"
                cases.append(
                    dict(
                        family=family,
                        group=group,
                        seed=seed,
                        alpha=chosen["alpha"],
                        name=name,
                        model_id=model_id(family, seed, group, chosen["alpha"]),
                    )
                )
    cases.append(
        dict(
            family="constant",
            group="constant",
            seed=None,
            alpha=None,
            name="constant",
            model_id="constant",
        )
    )
    assert len(cases) == 97
    return cases


def evaluate_stage(run, data, source):
    marker = js(run / "final_complete.json")
    assert marker["source_lock_sha256"] == source and marker["selection_sha256"] == sha(
        run / "selections.json"
    )
    selected = js(run / "selections.json")
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        bank = nz(run / "final" / role / "bank/models.npz")
        xx = [affine_features(data["color"][rows], s["dose"], s["anchor"]) for s in grid()]
        for case in evaluation_cases(selected, role):
            model = unpack(bank, case["model_id"])
            pred = np.stack([predict(model, x) for x in xx])
            scores = np.stack([gate_score(model, x) for x in xx]) if "gate_beta" in model else None
            transforms, doses = summaries(
                pred, data["target"][rows], data["patient"][rows], data["device"][rows], scores
            )
            mp = run / "selected" / role / f"{case['name']}.npz"
            pp = run / "evaluated" / role / f"{case['name']}.npz"
            atomic_npz(mp, model)
            atomic_npz(
                pp,
                dict(
                    prediction=pred,
                    row_indices=rows,
                    **({"scores": scores} if scores is not None else {}),
                ),
            )
            records.append(
                dict(
                    role=role,
                    **case,
                    numeric_bytes=sum(v.nbytes for v in model.values()),
                    archive_bytes=mp.stat().st_size,
                    input_dimensions=len(model["x_mean"]) if "x_mean" in model else 0,
                    kernel_dimensions=model["centers"].shape[1] if "centers" in model else 0,
                    actual_centers=len(model["centers"]) if "centers" in model else 0,
                    active_gate="gate_beta" in model,
                    model_sha256=sha(mp),
                    prediction_sha256=sha(pp),
                    metrics=metrics(
                        pred[0], data["target"][rows], data["patient"][rows], data["site"][rows]
                    ),
                    transforms=transforms,
                    doses=doses,
                )
            )
        print(f"EVALUATED {role}:97 models x33 transforms", flush=True)
    assert len(records) == 291
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=sha(run / "selections.json"),
            records=records,
            evidence="Reused TRAIN only; input feature ablation, no phone-face validation",
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
    start = time.perf_counter()
    source = lock_sources(args.run, args.cache)
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
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
                wall_seconds=time.perf_counter() - start,
                pid=os.getpid(),
                exit_code=0,
                scope="Read/fit/select/persist/evaluate excluding imports/audit/runtime; REUSE logs disclose reuse",
            ),
        )


if __name__ == "__main__":
    main()
