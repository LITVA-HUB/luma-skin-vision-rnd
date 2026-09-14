"""Frozen inherited H settings, whole-person teachers, matched C heads and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from chromaseed_affine_audit import exact
from chromaseed_crossfit import ARMS, LOSSES, SEEDS, fit_bank
from chromaseed_gate_stability import affine_features, gate_score, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_hybrid import predict
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/runs/chromaseed_hybrid_v1"


def freeze(run, cache):
    assert cache.name == "train.npz" and sha(cache) == CACHE_HASH
    ver = ROOT / "docs/benchmarks/chromaseed_hybrid_v1/verification.json"
    assert sha(ver) == "53883c8debdd447f0237640384439df2a886743b21152e7473cb7d959e242daa"
    old, inherited = js(ver), js(PARENT / "source_lock.json")
    for p, h in {
        **old["artifact_sha256"],
        **old["tested_sources"],
        **old["postprocess_sources"],
        **inherited["sources"],
        **inherited["input_sha256"],
    }.items():
        assert sha(ROOT / p) == h, p
    sources = dict(inherited["sources"])
    for p in (
        "scripts/chromaseed_crossfit.py",
        "scripts/chromaseed_crossfit_train.py",
        "tests/test_chromaseed_crossfit.py",
        "docs/research/chromaseed_crossfit_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    paths = [ver] + [PARENT / p for p in ("source_lock.json", "selections.json", "results.json")]
    for rec in js(PARENT / "results.json")["records"]:
        for area, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
            p = PARENT / area / rec["role"] / f"{rec['name']}.npz"
            assert sha(p) == rec[field]
            paths.append(p)
    paths += [
        ROOT / p
        for p in (
            "scripts/skin_crossfit_correction.py",
            "docs/research/skin_crossfit_correction_protocol_v1.md",
            "docs/research/skin_crossfit_correction_next_decision.md",
        )
    ]
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    assert len(sources) == 68 and len(inputs) == 229
    prior = js(PARENT / "selections.json")
    settings = {
        role: {
            loss: {
                kind: {
                    k: entry[loss][kind]["selected"][k] for k in ("kind", "alpha", "rho", "power")
                }
                for kind in ("uniform", "support")
            }
            for loss in LOSSES
        }
        for role, entry in prior["roles"].items()
    }
    value = json.loads(
        json.dumps(
            dict(
                sources=sources,
                input_sha256=inputs,
                cache_sha256=CACHE_HASH,
                settings=settings,
                seeds=SEEDS,
                arms=ARMS,
                rank=128,
                teacher_alpha=0.1,
                teacher_exclusion="one whole person",
                numpy=np.__version__,
                scipy=scipy.__version__,
                python=platform.python_version(),
                threads=1,
                transforms=grid(),
            )
        )
    )
    path = run / "source_lock.json"
    if path.exists():
        assert js(path) == value, "frozen C sources changed"
    else:
        write_json(path, value)
    selection = dict(
        source_lock_sha256=sha(path),
        parent_H_selection_sha256=sha(PARENT / "selections.json"),
        settings=settings,
        new_C_hyperparameter_selection=False,
    )
    sp = run / "frozen_settings.json"
    if sp.exists():
        assert js(sp) == selection
    else:
        write_json(sp, selection)
    return sha(path), settings


def train(run, data, source, settings):
    total = time.perf_counter()
    parent = js(PARENT / "results.json")
    for role, (fit, _) in roles(data["patient"], data["device"]).items():
        directory, rows = run / "banks" / role, np.flatnonzero(fit)
        if (directory / "receipt.json").exists():
            rec = js(directory / "receipt.json")
            assert rec["source_lock_sha256"] == source and rec["settings_sha256"] == sha(
                run / "frozen_settings.json"
            )
            for p, h in rec["files"].items():
                assert sha(directory / p) == h
            print(f"REUSE C bank {role}", flush=True)
            continue
        write_json(
            run / "progress.json",
            dict(status="fitting", role=role, pid=os.getpid(), updated_unix=time.time()),
        )
        current, teachers, tables, rec = fit_bank(
            *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
            settings[role],
        )
        models, cases = {}, []
        for p in (p for p in parent["records"] if p["role"] == role):
            m = nz(PARENT / "selected" / role / f"{p['name']}.npz")
            if p["kind"] == "raw":
                exact(m, current[f"{p['loss']}_raw_s{p['seed']}"])
                m = current[f"{p['loss']}_raw_s{p['seed']}"]
            name = "h_" + p["name"]
            models[name] = m
            cases.append(
                dict(
                    origin="H",
                    parent_name=p["name"],
                    family="h_" + p["family"],
                    name=name,
                    **{
                        k: p[k] for k in ("loss", "kind", "seed", "alpha", "rho", "power") if k in p
                    },
                )
            )
        for arm in ARMS:
            for loss in LOSSES:
                for kind, setting in settings[role][loss].items():
                    for seed in SEEDS:
                        name = f"{arm}_{loss}_{kind}_s{seed}"
                        models[name] = current[name]
                        cases.append(
                            dict(
                                origin="C",
                                family=f"{arm}_{loss}_{kind}",
                                name=name,
                                loss=loss,
                                arm=arm,
                                seed=seed,
                                **setting,
                            )
                        )
        assert len(models) == 61 and len(cases) == 61
        for case in cases:
            case["numeric_bytes"] = sum(v.nbytes for v in models[case["name"]].values())
        tables["row_indices"] = rows
        atomic_npz(directory / "models.npz", flatten(models))
        atomic_npz(directory / "teachers.npz", flatten(teachers))
        atomic_npz(directory / "tables.npz", tables)
        diagnostics = []
        for loss in LOSSES:
            for seed in SEEDS:
                full = tables[f"full__{loss}_s{seed}"]
                for arm in ("full", *ARMS):
                    native = tables[f"{arm}__{loss}_s{seed}"]
                    error = data["target"][rows] - native
                    diagnostics.append(
                        dict(
                            loss=loss,
                            seed=seed,
                            arm=arm,
                            metrics=metrics(
                                native,
                                data["target"][rows],
                                data["patient"][rows],
                                data["site"][rows],
                            ),
                            mean_signed_native_residual=error.mean(0).tolist(),
                            native_residual_rms=float(np.sqrt(np.mean(error**2))),
                            full_backbone_discrepancy_rms=float(
                                np.sqrt(np.mean((native - full) ** 2))
                            ),
                        )
                    )
        write_json(
            directory / "receipt.json",
            dict(
                **rec,
                source_lock_sha256=source,
                settings_sha256=sha(run / "frozen_settings.json"),
                cases=cases,
                supervision_diagnostics=diagnostics,
                teacher_names=list(teachers),
                files={p: sha(directory / p) for p in ("models.npz", "teachers.npz", "tables.npz")},
            ),
        )
        print(
            f"FIT C {role}: {rec['teacher_subsets']} excluded-person subsets, {len(teachers)} teachers,24 new corrections,37 H controls",
            flush=True,
        )
    write_json(
        run / "fit_complete.json",
        dict(
            source_lock_sha256=source,
            settings_sha256=sha(run / "frozen_settings.json"),
            banks=3,
            teacher_models=252,
            deployed_models=183,
            wall_seconds=time.perf_counter() - total,
        ),
    )


def evaluate(run, data, source):
    assert js(run / "fit_complete.json")["settings_sha256"] == sha(run / "frozen_settings.json")
    parent = {(r["role"], r["name"]): r for r in js(PARENT / "results.json")["records"]}
    records = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(held)
        rec, bank = (
            js(run / "banks" / role / "receipt.json"),
            nz(run / "banks" / role / "models.npz"),
        )
        xx = [affine_features(data["color"][rows], s["dose"], s["anchor"]) for s in grid()]
        for case in rec["cases"]:
            m = unpack(bank, case["name"])
            output = np.stack([predict(m, x) for x in xx])
            gates = None if "gate_beta" not in m else np.stack([gate_score(m, x) for x in xx])
            tt, dd = summaries(
                output, data["target"][rows], data["patient"][rows], data["device"][rows], gates
            )
            if case["origin"] == "H":
                expected = nz(PARENT / "evaluated" / role / f"{case['parent_name']}.npz")
                np.testing.assert_array_equal(rows, expected["row_indices"])
                np.testing.assert_array_equal(output, expected["prediction"])
                assert case["numeric_bytes"] == parent[(role, case["parent_name"])]["numeric_bytes"]
            mp, pp = (
                run / "selected" / role / f"{case['name']}.npz",
                run / "evaluated" / role / f"{case['name']}.npz",
            )
            atomic_npz(mp, m)
            atomic_npz(
                pp,
                dict(
                    prediction=output,
                    row_indices=rows,
                    **({} if gates is None else dict(scores=gates)),
                ),
            )
            records.append(
                dict(
                    role=role,
                    **case,
                    model_sha256=sha(mp),
                    prediction_sha256=sha(pp),
                    archive_bytes=mp.stat().st_size,
                    metrics=metrics(
                        output[0], data["target"][rows], data["patient"][rows], data["site"][rows]
                    ),
                    transforms=tt,
                    doses=dd,
                )
            )
        print(
            f"EVALUATED C {role}:61 models x33 fixed transforms,37 H outputs exactly unchanged",
            flush=True,
        )
    write_json(
        run / "results.json",
        dict(
            source_lock_sha256=source,
            settings_sha256=sha(run / "frozen_settings.json"),
            records=records,
            evidence="fixed-H-configuration residual-supervision ablation on reused original TRAIN; no independent phone-face evidence",
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
    parser.add_argument("stage", choices=("run", "fit", "evaluate"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    source, settings = freeze(args.run, args.cache)
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    if args.stage in ("run", "fit"):
        train(args.run, data, source, settings)
    if args.stage in ("run", "evaluate"):
        evaluate(args.run, data, source)
    if args.stage == "run":
        write_json(
            args.run / "workflow.json",
            dict(
                source_lock_sha256=source,
                wall_seconds=time.perf_counter() - start,
                scope="read/freeze/fit/persist/evaluate; excludes imports/audit/profiling; logs identify any reuse",
            ),
        )


if __name__ == "__main__":
    main()
