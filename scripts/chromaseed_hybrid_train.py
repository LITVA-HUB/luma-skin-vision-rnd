"""H frozen shared-center training, person-held-out selection and fixed stress evaluation."""

from __future__ import annotations

import argparse
import json
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
from chromaseed_hybrid import (
    ALPHAS,
    KINDS,
    LOSSES,
    POWERS,
    RHOS,
    SEEDS,
    choose,
    fit_bank,
    model_id,
    predict,
    settings,
)
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_projection import model_id as x_id
from chromaseed_projection_train import parent_bank, score
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
PARENT_A = ROOT / "experiments/runs/chromaseed_affine_v1"
PARENT_X = ROOT / "experiments/runs/chromaseed_projection_v1"


def lock_sources(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    ver = ROOT / "docs/benchmarks/chromaseed_projection_v1/verification.json"
    assert sha(ver) == "537014c2d803f27bb9a72f93d260c2b0d1a830a9732fc4037bfe3dc202a98153"
    old = js(ver)
    assert old["passed"]
    for p, h in {
        **old["artifact_sha256"],
        **old["postprocess_sources"],
        **old["tested_sources"],
    }.items():
        assert sha(ROOT / p) == h, p
    inherited = js(PARENT_X / "source_lock.json")
    for p, h in {**inherited["sources"], **inherited["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    sources = dict(inherited["sources"])
    for p in (
        "scripts/chromaseed_hybrid.py",
        "scripts/chromaseed_hybrid_numpy.py",
        "scripts/chromaseed_hybrid_train.py",
        "tests/test_chromaseed_hybrid.py",
        "docs/research/chromaseed_hybrid_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    paths = [ver]
    for parent in (PARENT_A, PARENT_X):
        paths += [parent / p for p in ("source_lock.json", "selections.json", "results.json")]
        for directory in sorted((parent / "inner").glob("*/fold*")) + sorted(
            (parent / "final").glob("*/bank")
        ):
            paths += [directory / "models.npz", directory / "receipt.json"]
            if (directory / "oof.npz").exists():
                paths.append(directory / "oof.npz")
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    assert len(sources) == 64 and len(inputs) == 73
    value = json.loads(
        json.dumps(
            dict(
                sources=sources,
                input_sha256=inputs,
                cache_sha256=CACHE_HASH,
                losses=LOSSES,
                kinds=KINDS,
                alphas=ALPHAS,
                seeds=SEEDS,
                rhos=RHOS,
                powers=POWERS,
                dimension=16,
                shrinkage=0.5,
                raw_alpha=0.1,
                tie_tolerance=1e-5,
                rank=128,
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
        assert js(path) == value, "frozen H sources changed"
    else:
        write_json(path, value)
    return sha(path)


def controls(models, receipt, data, rows, role, stage, sub):
    a = parent_bank(PARENT_A, role, stage, sub, rows)
    x = parent_bank(PARENT_X, role, stage, sub, rows)
    for loss in LOSSES:
        for seed in SEEDS:
            raw = models[model_id(loss, seed, "raw")]
            expected = unpack(a, a_id(loss + "_joint_soft", seed, 0.1, 0.0))
            assert set(raw) == set(expected)
            for key in raw:
                np.testing.assert_array_equal(raw[key], expected[key])
            prior = x_id(loss + "_joint_soft", seed, "d16_t05", 0.1)
            name = f"x_fixed_{loss}_s{seed}"
            models[name] = unpack(x, prior)
            receipt["models"][name] = dict(
                loss=loss, seed=seed, kind="reference", prior_model_id=prior
            )
    models["constant"] = dict(
        constant_lab=np.average(
            data["target"][rows],
            axis=0,
            weights=weights_for(data["patient"][rows], data["site"][rows]),
        ).astype(np.float32)
    )
    receipt["models"]["constant"] = dict(loss=None, seed=None, kind="constant")
    receipt.update(
        exact_A_raw_payloads=6, imported_X_payloads=6, constant_fits=1, endpoint_aliases=18
    )


def run_bank(run, data, role, fit, query, fold, source):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    directory = run / stage / role / sub
    if valid_bank(directory, source):
        print(f"REUSE {stage}/{role}/{sub}", flush=True)
        return
    if query is not None:
        assert not set(data["patient"][fit]) & set(data["patient"][query])
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
    assert len(models) == 247
    for name, meta in receipt["models"].items():
        meta["numeric_bytes"] = sum(v.nbytes for v in models[name].values())
    atomic_npz(directory / "models.npz", flatten(models))
    files = {"models.npz": sha(directory / "models.npz")}
    if query is not None:
        saved = dict(row_indices=query)
        for name, m in models.items():
            saved["pred__" + name] = predict(m, data["color"][query])
        old = nz(PARENT_A / stage / role / sub / "oof.npz")
        xx = nz(PARENT_X / stage / role / sub / "oof.npz")
        np.testing.assert_array_equal(query, old["row_indices"])
        np.testing.assert_array_equal(query, xx["row_indices"])
        for loss in LOSSES:
            for seed in SEEDS:
                np.testing.assert_allclose(
                    saved["pred__" + model_id(loss, seed, "raw")],
                    old["pred__" + a_id(loss + "_joint_soft", seed, 0.1, 0.0)][0],
                    rtol=0,
                    atol=2e-8,
                )
                np.testing.assert_allclose(
                    saved[f"pred__x_fixed_{loss}_s{seed}"],
                    xx["pred__" + x_id(loss + "_joint_soft", seed, "d16_t05", 0.1)],
                    rtol=0,
                    atol=2e-8,
                )
        atomic_npz(directory / "oof.npz", saved)
        files["oof.npz"] = sha(directory / "oof.npz")
    write_json(
        directory / "receipt.json",
        dict(
            **receipt,
            source_lock_sha256=source,
            fit_rows_sha256=row_hash(fit),
            query_rows_sha256=None if query is None else row_hash(query),
            n_fit_people=len(np.unique(data["patient"][fit])),
            files=files,
        ),
    )
    print(f"FIT {stage}/{role}/{sub}:247 records,78 solutions,6 exact A", flush=True)


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
            readouts=741 if final else 2223,
            selection_sha256=sha(run / "selections.json") if final else None,
        ),
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
        for loss in LOSSES:
            selected["roles"][role][loss] = {}
            for kind in KINDS:
                candidates = []
                for setting in settings(kind):
                    names = [model_id(loss, s, **setting) for s in SEEDS]
                    ss = [scores(name) for name in names]
                    candidates.append(
                        dict(
                            **setting,
                            clean=float(np.mean([s["clean"] for s in ss])),
                            p90=float(np.mean([s["p90"] for s in ss])),
                            numeric_bytes=max(sizes[n] for n in names),
                            seed_scores=ss,
                        )
                    )
                selected["roles"][role][loss][kind] = dict(
                    candidates=candidates, selected=choose(candidates)
                )
        refs = {}
        for loss in LOSSES:
            ss = [scores(f"x_fixed_{loss}_s{s}") for s in SEEDS]
            refs[f"x_fixed_{loss}"] = dict(
                clean=float(np.mean([s["clean"] for s in ss])),
                p90=float(np.mean([s["p90"] for s in ss])),
                seed_scores=ss,
            )
        refs["constant"] = scores("constant")
        selected["references"][role] = refs
    path = run / "selections.json"
    if path.exists():
        assert js(path) == selected
    else:
        write_json(path, selected)
    print("FROZEN30 choices from258 candidate score records", flush=True)


def cases(selected, role):
    result = []
    for loss in LOSSES:
        for family in KINDS:
            c = selected["roles"][role][loss][family]["selected"]
            setting = {k: c[k] for k in ("kind", "alpha", "rho", "power")}
            for seed in SEEDS:
                result.append(
                    dict(
                        loss=loss,
                        family=f"{loss}_{family}",
                        policy="inner",
                        seed=seed,
                        **setting,
                        name=f"{loss}_{family}_s{seed}",
                        model_id=model_id(loss, seed, **setting),
                    )
                )
    for loss in LOSSES:
        for seed in SEEDS:
            name = f"x_fixed_{loss}_s{seed}"
            result.append(
                dict(
                    loss=loss,
                    family=f"x_fixed_{loss}",
                    policy="reference",
                    seed=seed,
                    kind="reference",
                    name=name,
                    model_id=name,
                )
            )
    result.append(
        dict(
            loss=None,
            family="constant",
            policy="reference",
            seed=None,
            kind="constant",
            name="constant",
            model_id="constant",
        )
    )
    assert len(result) == 37
    return result


def evaluate_stage(run, data, source):
    assert js(run / "final_complete.json")["selection_sha256"] == sha(run / "selections.json")
    selected, records = js(run / "selections.json"), []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        bank = nz(run / "final" / role / "bank/models.npz")
        xx = [affine_features(data["color"][rows], s["dose"], s["anchor"]) for s in grid()]
        for case in cases(selected, role):
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
                    active_gate="gate_beta" in m,
                    active_hybrid="hybrid_mode" in m,
                    model_sha256=sha(mp),
                    prediction_sha256=sha(pp),
                    metrics=metrics(
                        pred[0], data["target"][rows], data["patient"][rows], data["site"][rows]
                    ),
                    transforms=tt,
                    doses=dd,
                )
            )
        print(f"EVALUATED {role}:37 models x33 fixed transforms", flush=True)
    assert len(records) == 111
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=source,
            selection_sha256=sha(run / "selections.json"),
            records=records,
            evidence="reused TRAIN; shared original landmark indices; no ordinary-phone facial validation",
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
                scope="read/fit/select/persist/evaluate excluding imports/audit/runtime; logs identify any reuse",
            ),
        )


if __name__ == "__main__":
    main()
