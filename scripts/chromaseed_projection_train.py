"""Frozen X representation banks, inner quality/compact policies and fixed-dose finals."""

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
from chromaseed_gate_stability import affine_features, gate_score, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gated import model_id as g_id
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_projection import (
    ALPHAS,
    FAMILIES,
    POLICIES,
    REPRESENTATIONS,
    SEEDS,
    choose,
    fit_bank,
    model_id,
    predict,
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

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
PARENT_A = ROOT / "experiments/runs/chromaseed_affine_v1"
PARENT_G = ROOT / "experiments/runs/chromaseed_gated_v1"
CONTROLS = (
    "g_norm_soft",
    "g_perceptual_soft",
    "a_norm_joint_guarded",
    "a_perceptual_joint_guarded",
)


def lock_sources(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    ver = ROOT / "docs/benchmarks/chromaseed_affine_v1/verification.json"
    assert sha(ver) == "91a348711bbe95dba917a6b31c34e5c78d94c6113737a33f0a93ba4fcc5ec646"
    old = js(ver)
    assert old["passed"]
    for p, h in {
        **old["artifact_sha256"],
        **old["postprocess_sources"],
        **old["tested_sources"],
    }.items():
        assert sha(ROOT / p) == h, p
    inherited = js(PARENT_A / "source_lock.json")
    sources = dict(inherited["sources"])
    for p, h in {**sources, **inherited["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    for p in (
        "scripts/chromaseed_projection.py",
        "scripts/chromaseed_projection_numpy.py",
        "scripts/chromaseed_projection_train.py",
        "tests/test_chromaseed_projection.py",
        "docs/research/chromaseed_projection_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    paths = [ver]
    for parent in (PARENT_A, PARENT_G):
        paths += [parent / p for p in ("source_lock.json", "selections.json", "results.json")]
        for directory in sorted((parent / "inner").glob("*/fold*")) + sorted(
            (parent / "final").glob("*/bank")
        ):
            paths += [directory / "models.npz", directory / "receipt.json"]
            if (directory / "oof.npz").exists():
                paths.append(directory / "oof.npz")
        for rec in js(parent / "results.json")["records"]:
            keep = (
                (rec["family"] in ("norm_soft", "perceptual_soft"))
                if parent == PARENT_G
                else (
                    rec["family"] in ("norm_joint_soft", "perceptual_joint_soft")
                    and rec["policy"] == "guarded"
                )
            )
            if keep:
                name = f"{rec['family']}_s{rec['seed']}" if parent == PARENT_G else rec["name"]
                for area, field in (
                    ("selected", "model_sha256"),
                    ("evaluated", "prediction_sha256"),
                ):
                    p = parent / area / rec["role"] / f"{name}.npz"
                    assert sha(p) == rec[field]
                    paths.append(p)
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    assert len(sources) == 59 and len(inputs) == 145, (len(sources), len(inputs))
    import json

    value = json.loads(
        json.dumps(
            dict(
                sources=sources,
                input_sha256=inputs,
                cache_sha256=CACHE_HASH,
                representations=REPRESENTATIONS,
                families=FAMILIES,
                policies=POLICIES,
                seeds=SEEDS,
                alphas=ALPHAS,
                controls=CONTROLS,
                rank=128,
                clean_allowance=0.05,
                p90_allowance=0.10,
                final_settings=grid(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                python=platform.python_version(),
                threads=1,
            )
        )
    )
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == value, "frozen X sources changed"
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


def controls(models, receipt, data, rows, role, stage, sub):
    a = parent_bank(PARENT_A, role, stage, sub, rows)
    g = parent_bank(PARENT_G, role, stage, sub, rows)
    aa, gg = (
        js(PARENT_A / "selections.json")["roles"][role],
        js(PARENT_G / "selections.json")["roles"][role],
    )
    raw_count = 0
    for family in FAMILIES:
        for seed in SEEDS:
            for alpha in ALPHAS:
                m = models[model_id(family, seed, "raw", alpha)]
                original = unpack(a, a_id(family, seed, alpha, 0.0))
                assert set(m) == set(original)
                for key in m:
                    np.testing.assert_array_equal(m[key], original[key])
                raw_count += 1
    for base in ("norm", "perceptual"):
        ga, ac = gg[base + "_soft"]["selected"], aa[base + "_joint_soft"]["policies"]["guarded"]
        for seed in SEEDS:
            for family, prior_id, bank in (
                (f"g_{base}_soft", g_id(base + "_soft", seed, ga["residual_lambda"], ga["rho"]), g),
                (
                    f"a_{base}_joint_guarded",
                    a_id(base + "_joint_soft", seed, ac["alpha"], ac["eta"]),
                    a,
                ),
            ):
                name = f"{family}_s{seed}"
                models[name] = unpack(bank, prior_id)
                receipt["models"][name] = dict(
                    family=family, seed=seed, reference=True, prior_model_id=prior_id
                )
    models["constant"] = dict(
        constant_lab=np.average(
            data["target"][rows],
            axis=0,
            weights=weights_for(data["patient"][rows], data["site"][rows]),
        ).astype(np.float32)
    )
    receipt["models"]["constant"] = dict(family="constant", seed=None, reference=True)
    receipt.update(
        exact_A_raw_payloads=raw_count,
        imported_G_payloads=6,
        imported_A_payloads=6,
        constant_fits=1,
    )


def run_bank(run, data, role, fit, queries, fold, source):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    directory = run / stage / role / sub
    if valid_bank(directory, source):
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
    controls(models, receipt, data, fit, role, stage, sub)
    assert len(models) == 481
    aliases = 0
    for name, meta in receipt["models"].items():
        meta["numeric_bytes"] = sum(v.nbytes for v in models[name].values())
        if meta.get("fallback"):
            other = models[
                model_id(
                    meta["family"].replace("joint_soft", "static"),
                    meta["seed"],
                    meta["representation"],
                    meta["alpha"],
                )
            ]
            assert set(models[name]) == set(other)
            for k in other:
                np.testing.assert_array_equal(models[name][k], other[k])
            aliases += 1
    atomic_npz(directory / "models.npz", flatten(models))
    files = {"models.npz": sha(directory / "models.npz")}
    if queries is not None:
        saved = dict(row_indices=queries)
        for name, m in models.items():
            saved["pred__" + name] = predict(m, data["color"][queries])
        # Original A eta0 inner outputs are exact-model controls as well.
        old = nz(PARENT_A / "inner" / role / sub / "oof.npz")
        np.testing.assert_array_equal(queries, old["row_indices"])
        for family in FAMILIES:
            for seed in SEEDS:
                for alpha in ALPHAS:
                    np.testing.assert_allclose(
                        saved["pred__" + model_id(family, seed, "raw", alpha)],
                        old["pred__" + a_id(family, seed, alpha, 0)][0],
                        atol=2e-8,
                        rtol=0,
                    )
        atomic_npz(directory / "oof.npz", saved)
        files["oof.npz"] = sha(directory / "oof.npz")
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
    write_json(directory / "receipt.json", receipt)
    print(
        f"FIT {stage}/{role}/{sub}:481 stored,{receipt['new_coefficient_solutions']} solved,{aliases} aliases,36 exact A",
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
            readouts=1443 if final else 4329,
            selection_sha256=sha(run / "selections.json") if final else None,
        ),
    )


def score(pred, target, person):
    error = delta_e00(pred, target)
    return dict(
        clean=float(np.mean([error[person == p].mean() for p in np.unique(person)])),
        p90=float(np.quantile(error, 0.9)),
    )


def select_stage(run, data, source):
    assert js(run / "inner_complete.json")["source_lock_sha256"] == source
    selected = dict(source_lock_sha256=source, roles={}, references={})
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

        def scores(name):
            return score(merged[name], data["target"][rows], data["patient"][rows])

        selected["roles"][role] = {}
        for family in FAMILIES:
            candidates = []
            for order, rep in enumerate(REPRESENTATIONS):
                for alpha in ALPHAS:
                    names = [model_id(family, s, rep["name"], alpha) for s in SEEDS]
                    sm = [scores(name) for name in names]
                    candidates.append(
                        dict(
                            family=family,
                            representation=rep["name"],
                            order=order,
                            alpha=alpha,
                            clean=float(np.mean([s["clean"] for s in sm])),
                            p90=float(np.mean([s["p90"] for s in sm])),
                            numeric_bytes=max(sizes[name] for name in names),
                            seed_scores=sm,
                        )
                    )
            raw = min(
                (r for r in candidates if r["representation"] == "raw"),
                key=lambda r: (r["clean"], r["p90"], -r["alpha"]),
            )
            selected["roles"][role][family] = dict(
                candidates=candidates,
                raw_anchor=raw,
                policies={p: choose(candidates, p) for p in POLICIES},
                clean_allowance=0.05,
                p90_allowance=0.10,
            )
        refs = {}
        for family in CONTROLS:
            sm = [scores(f"{family}_s{s}") for s in SEEDS]
            refs[family] = dict(
                clean=float(np.mean([s["clean"] for s in sm])),
                p90=float(np.mean([s["p90"] for s in sm])),
                seed_scores=sm,
            )
        refs["constant"] = scores("constant")
        selected["references"][role] = refs
    path = run / "selections.json"
    if path.exists():
        assert js(path) == selected
    else:
        write_json(path, selected)
    print("FROZEN24 choices from468 candidate score pairs;15 fixed reference groups", flush=True)


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
                        representation=chosen["representation"],
                        name=f"{family}_{policy}_s{seed}",
                        model_id=model_id(family, seed, chosen["representation"], chosen["alpha"]),
                    )
                )
    for family in CONTROLS:
        for seed in SEEDS:
            name = f"{family}_s{seed}"
            cases.append(
                dict(family=family, policy="reference", seed=seed, name=name, model_id=name)
            )
    cases.append(
        dict(family="constant", policy="reference", seed=None, name="constant", model_id="constant")
    )
    assert len(cases) == 37
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
            m = unpack(bank, case["model_id"])
            pred = np.stack([predict(m, x) for x in xx])
            ss = np.stack([gate_score(m, x) for x in xx]) if "gate_beta" in m else None
            tt, dd = summaries(
                pred, data["target"][rows], data["patient"][rows], data["device"][rows], ss
            )
            mp, pp = (
                run / "selected" / role / f"{case['name']}.npz",
                run / "evaluated" / role / f"{case['name']}.npz",
            )
            atomic_npz(mp, m)
            atomic_npz(
                pp,
                dict(
                    prediction=pred, row_indices=rows, **({"scores": ss} if ss is not None else {})
                ),
            )
            records.append(
                dict(
                    role=role,
                    **case,
                    numeric_bytes=sum(v.nbytes for v in m.values()),
                    archive_bytes=mp.stat().st_size,
                    actual_centers=0 if "constant_lab" in m else len(m["centers"]),
                    latent_dimension=0 if "constant_lab" in m else m["centers"].shape[1],
                    active_gate="gate_beta" in m,
                    model_sha256=sha(mp),
                    prediction_sha256=sha(pp),
                    metrics=metrics(
                        pred[0], data["target"][rows], data["patient"][rows], data["site"][rows]
                    ),
                    transforms=tt,
                    doses=dd,
                )
            )
        print(f"EVALUATED {role}:37 final models x33 transforms", flush=True)
    assert len(records) == 111
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=sha(run / "selections.json"),
            records=records,
            evidence="historically reused TRAIN roles; fit-only unsupervised projections; no ordinary-phone facial validation",
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
                scope="read/fit/select/persist/evaluate excluding imports/audit/runtime; logs identify any reused banks",
            ),
        )


if __name__ == "__main__":
    main()
