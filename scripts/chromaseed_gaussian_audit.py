"""Independent TG bank/OOF/curve audit and full scalar retraining of selected traces."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact, summaries
from chromaseed_feature_groups_audit import compare
from chromaseed_feature_groups_audit import direct as fg_direct
from chromaseed_feature_groups_numpy import Predictor as FGPredictor
from chromaseed_gate_stability_audit import close, transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_numpy import Predictor
from chromaseed_gaussian_reference import normalization, train_reference
from chromaseed_gaussian_reference import predict as ref_predict
from chromaseed_kernel_audit import js, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
FG = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
SOURCE = "c5f2d9f4a4a9111b4da1a5b238c0d48ecd6c851a808efb272d66e5c92cebca62"
METHODS = ("adam", "tagi_diag", "tagi_full3")
GROUPS = ("raw36", "mean3")
SEEDS = (17, 29, 43)
EPOCHS = (1, 4, 16, 64)
PARAMETERS = {
    "adam": (0.0003, 0.001, 0.003),
    "tagi_diag": (0.1, 0.3, 1.0),
    "tagi_full3": (0.1, 0.3, 1.0),
}
SETTINGS = [(0.0, np.zeros(3))] + [
    (d / 255, np.array(a)) for d in (1, 4, 16, 64) for a in itertools.product((0.0, 1.0), repeat=3)
]


def ident(method, group, seed, parameter, epoch):
    return f"{method}_{group}_s{seed}_h{PARAMETERS[method].index(parameter)}_e{epoch}"


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def direct(model, x):
    return ref_predict(model, x) if "w1" in model else fg_direct(model, x)


def actual_consumer(model):
    return Predictor(model) if "w1" in model else FGPredictor(model)


def scoring(pred, y, person):
    error = delta_e00(pred, y)
    return dict(
        clean=float(np.mean([np.mean(error[person == p]) for p in np.unique(person)])),
        p90=float(np.percentile(error, 90)),
    )


def expected_names(role, selection, final):
    names = {}
    for method in METHODS:
        for group in GROUPS:
            parameters = (
                (selection["roles"][role][method][group]["selected"]["parameter"],)
                if final
                else PARAMETERS[method]
            )
            for seed, parameter, epoch in itertools.product(SEEDS, parameters, EPOCHS):
                names[ident(method, group, seed, parameter, epoch)] = dict(
                    method=method,
                    group=group,
                    seed=seed,
                    parameter=parameter,
                    parameter_index=PARAMETERS[method].index(parameter),
                    epoch=epoch,
                    origin="fit",
                )
    for group, seed in itertools.product(GROUPS, SEEDS):
        names[f"fg_norm_static_{group}_s{seed}"] = dict(
            method="fg_norm_static",
            group=group,
            seed=seed,
            parameter=0.1,
            parameter_index=0,
            epoch=None,
            origin="FG",
            source_model_id=f"norm_static_{group}_s{seed}_a0",
        )
    names["constant"] = dict(
        method="constant",
        group="constant",
        seed=None,
        parameter=None,
        parameter_index=None,
        epoch=None,
        origin="FG",
        source_model_id="constant",
    )
    return names


def bank_check(run, data, fit, query, role, fold, selection, checks, maxima):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    path, parent_path = run / stage / role / sub, FG / stage / role / sub
    rec, parent = js(path / "receipt.json"), js(parent_path / "receipt.json")
    assert rec["source_lock_sha256"] == SOURCE
    assert rec["selection_sha256"] == (sha(run / "selections.json") if fold is None else None)
    assert rec["fit_rows_sha256"] == parent["fit_rows_sha256"] == row_hash(fit)
    assert rec["query_rows_sha256"] == (None if query is None else row_hash(query))
    assert rec["n_fit_rows"] == len(fit) and rec["n_fit_people"] == len(
        np.unique(data["patient"][fit])
    )
    for directory, receipt in ((path, rec), (parent_path, parent)):
        for p, h in receipt["files"].items():
            assert sha(directory / p) == h
    expected = expected_names(role, selection, fold is None)
    assert set(expected) == set(rec["models"]) and rec["imported_FG_models"] == 7
    arrays, parent_arrays = nz(path / "models.npz"), nz(parent_path / "models.npz")
    assert len(rec["trajectories"]) == (18 if fold is None else 54)
    preparations = {g: normalization(data["color"][fit], data["target"][fit], g)[0] for g in GROUPS}
    order_hashes = {}
    for seed in SEEDS:
        rng, digest = np.random.default_rng(seed + 600017), hashlib.sha256()
        for epoch in range(1, 65):
            digest.update(np.asarray(rng.permutation(len(fit)), np.int64).tobytes())
            if epoch in EPOCHS:
                order_hashes[(seed, epoch)] = digest.hexdigest()
    for trace in rec["trajectories"]:
        assert trace["n_fit_rows"] == len(fit)
        size = 2563 if trace["group"] == "raw36" else 451
        assert trace["parameter_count"] == size
        assert trace["training_parameter_state_bytes"] == size * (
            24 if trace["method"] == "adam" else 16
        )
        assert [c["epoch"] for c in trace["checkpoints"]] == list(EPOCHS)
        last_floor, last_negative, last_time = 0, 0, 0
        for cp in trace["checkpoints"]:
            assert cp["examples_seen"] == cp["epoch"] * len(fit)
            assert cp["order_sha256"] == order_hashes[(trace["seed"], cp["epoch"])]
            assert (
                cp["floored_variance_updates"] >= last_floor
                and cp["negative_variance_updates"] >= last_negative
            )
            assert (
                np.isfinite(cp["block_elapsed_seconds"])
                and cp["block_elapsed_seconds"] >= last_time
            )
            last_floor, last_negative, last_time = (
                cp["floored_variance_updates"],
                cp["negative_variance_updates"],
                cp["block_elapsed_seconds"],
            )
            if trace["method"] == "adam":
                assert last_floor == last_negative == 0 and cp["minimum_variance"] is None
            else:
                assert cp["minimum_variance"] >= 1e-12 and np.isfinite(
                    cp["minimum_pre_floor_variance"]
                )
                if trace["method"] == "tagi_full3":
                    assert last_negative == 0
            checks["trajectory_checkpoints"] += 1
        checks["trajectories"] += 1
    assert rec["examples_seen_all_trajectories"] == len(rec["trajectories"]) * len(fit) * 64
    checks["example_updates"] += rec["examples_seen_all_trajectories"]
    models = {}
    for name, info in expected.items():
        saved = rec["models"][name]
        for k, v in info.items():
            assert saved[k] == v
        model = model_from(arrays, name)
        assert saved["numeric_bytes"] == sum(v.nbytes for v in model.values())
        if info["origin"] == "FG":
            exact(model, model_from(parent_arrays, info["source_model_id"]))
            checks["exact_FG_models"] += 1
            if name == "constant":
                wanted = np.average(
                    data["target"][fit],
                    axis=0,
                    weights=balanced(data["patient"][fit], data["site"][fit]),
                ).astype(np.float32)
                np.testing.assert_array_equal(model["constant_lab"], wanted)
        else:
            for k, v in preparations[info["group"]].items():
                np.testing.assert_array_equal(model[k], v)
            assert saved["numeric_bytes"] == (10564 if info["group"] == "raw36" else 1855)
            assert model["w1"].shape == (64, 36 if info["group"] == "raw36" else 3)
            trace = rec["trajectories"][saved["trajectory_index"]]
            assert all(trace[k] == info[k] for k in ("method", "group", "seed", "parameter"))
            Predictor(model)
            checks["new_checkpoints"] += 1
        models[name] = model
        checks["stored_models"] += 1
    if query is not None:
        assert not set(data["patient"][fit]) & set(data["patient"][query])
        oof, parent_oof = nz(path / "oof.npz"), nz(parent_path / "oof.npz")
        np.testing.assert_array_equal(oof["row_indices"], query)
        np.testing.assert_array_equal(parent_oof["row_indices"], query)
        assert set(oof) == {"row_indices"} | {"pred__" + n for n in models}
        for name, model in models.items():
            independent = direct(model, data["color"][query])
            saved = oof["pred__" + name]
            assert saved.shape == (len(query), 3)
            drift = float(np.max(np.abs(independent - saved)))
            maxima["oof_prediction"] = max(maxima["oof_prediction"], drift)
            assert drift <= 2e-8
            if expected[name]["origin"] == "FG":
                np.testing.assert_array_equal(
                    saved, parent_oof["pred__" + expected[name]["source_model_id"]]
                )
            checks["oof_rows"] += len(query)
    checks["banks"] += 1
    print(f"AUDIT {stage}/{role}/{sub}:payloads,normalizers,orders and outputs checked", flush=True)
    return models, rec


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, out, started = args.run, args.output, time.perf_counter()
    lock, selection, result = (
        js(run / p) for p in ("source_lock.json", "selections.json", "results.json")
    )
    assert (
        sha(run / "source_lock.json")
        == SOURCE
        == selection["source_lock_sha256"]
        == result["source_lock_sha256"]
    )
    assert result["selection_sha256"] == sha(run / "selections.json")
    assert js(run / "workflow.json")["exit_code"] == 0
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert len(lock["sources"]) == 79 and len(lock["input_sha256"]) == 42
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    checks = {
        k: 0
        for k in (
            "banks",
            "stored_models",
            "new_checkpoints",
            "exact_FG_models",
            "trajectories",
            "trajectory_checkpoints",
            "example_updates",
            "oof_rows",
            "candidate_scores",
            "choices",
            "clean_models",
            "clean_rows",
            "selected_models",
            "selected_rows",
            "transform_cases",
            "dose_cases",
            "scalar_refit_trajectories",
            "scalar_refit_checkpoints",
            "scalar_refit_clean_rows",
            "scalar_refit_stress_rows",
        )
    }
    maxima = {
        k: 0.0
        for k in (
            "oof_prediction",
            "direct_prediction",
            "consumer_prediction",
            "refit_parameter_absolute",
            "refit_prediction",
        )
    }
    refits, person_errors, variance_diagnostics = [], {}, []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        rows, queries = np.flatnonzero(fit), np.flatnonzero(held)
        folds = folds_for(data["patient"][rows], data["device"][rows])
        merged, seen = {}, []
        for fold in range(3):
            fit_rows, query_rows = rows[folds != fold], rows[folds == fold]
            _, rec = bank_check(
                run, data, fit_rows, query_rows, role, fold, selection, checks, maxima
            )
            variance_diagnostics += [
                dict(role=role, stage="inner", fold=fold, **t) for t in rec["trajectories"]
            ]
            saved = nz(run / "inner" / role / f"fold{fold}" / "oof.npz")
            ix = np.flatnonzero(folds == fold)
            seen.extend(ix.tolist())
            for name in rec["models"]:
                if name not in merged:
                    merged[name] = np.empty((len(rows), 3))
                merged[name][ix] = saved["pred__" + name]
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))
        for method in METHODS:
            for group in GROUPS:
                candidates = []
                for pindex, parameter in enumerate(PARAMETERS[method]):
                    for epoch in EPOCHS:
                        scores = [
                            scoring(
                                merged[ident(method, group, seed, parameter, epoch)],
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
                stored = selection["roles"][role][method][group]
                compare(stored["candidates"], candidates)
                winner = sorted(
                    candidates,
                    key=lambda v: (v["clean"], v["p90"], v["epoch"], v["parameter_index"]),
                )[0]
                compare(stored["selected"], winner)
                checks["candidate_scores"] += 12
                checks["choices"] += 1
        refs = {
            name: scoring(pred, data["target"][rows], data["patient"][rows])
            for name, pred in merged.items()
            if name.startswith("fg_") or name == "constant"
        }
        compare(selection["references"][role], refs)
        final_models, final_rec = bank_check(
            run, data, rows, None, role, None, selection, checks, maxima
        )
        variance_diagnostics += [
            dict(role=role, stage="final", fold=None, **t) for t in final_rec["trajectories"]
        ]
        x, y, p, s, c = (data[k][queries] for k in ("color", "target", "patient", "site", "device"))
        assert not set(p) & set(data["patient"][rows])
        xx = [transformed(x, d, a) for d, a in SETTINGS]
        curves = {r["name"]: r for r in result["curve_records"] if r["role"] == role}
        selected = {r["name"]: r for r in result["records"] if r["role"] == role}
        assert set(curves) == set(final_models) and len(curves) == 79 and len(selected) == 25
        expected_selected = {
            name
            for name, r in curves.items()
            if r["origin"] == "FG"
            or r["epoch"] == selection["roles"][role][r["method"]][r["group"]]["selected"]["epoch"]
        }
        assert set(selected) == expected_selected
        for name, rec in curves.items():
            mp, cp = run / "models" / role / f"{name}.npz", run / "curves" / role / f"{name}.npz"
            assert sha(mp) == rec["model_sha256"] and sha(cp) == rec["clean_prediction_sha256"]
            model, saved = nz(mp), nz(cp)
            exact(model, final_models[name])
            for k, v in final_rec["models"][name].items():
                assert rec[k] == v
            assert rec["selected_for_stress"] == (name in selected)
            assert rec["archive_bytes"] == mp.stat().st_size and rec["numeric_bytes"] == sum(
                v.nbytes for v in model.values()
            )
            np.testing.assert_array_equal(saved["row_indices"], queries)
            assert saved["prediction"].shape == (len(x), 3)
            independent = direct(model, x)
            actual = actual_consumer(model)
            deployed = np.array([actual(row) for row in x])
            for label, output in (
                ("direct_prediction", independent),
                ("consumer_prediction", deployed),
            ):
                drift = float(np.abs(output - saved["prediction"]).max())
                maxima[label] = max(maxima[label], drift)
                assert drift <= 2e-8
            metric, per_person = error_summary(independent, y, p, s)
            compare(rec["metrics"], metric)
            if name in selected:
                person_errors[(role, rec["method"], rec["group"], rec["seed"])] = per_person
            checks["clean_models"] += 1
            checks["clean_rows"] += len(x)
        for name, rec in selected.items():
            ep = run / "evaluated" / role / f"{name}.npz"
            assert sha(ep) == rec["prediction_sha256"]
            saved = nz(ep)
            np.testing.assert_array_equal(saved["row_indices"], queries)
            assert saved["prediction"].shape == (33, len(x), 3)
            model = final_models[name]
            output = np.stack([direct(model, z) for z in xx])
            consumer = actual_consumer(model)
            actual = np.array([[consumer(v) for v in z] for z in xx])
            for label, prediction in (
                ("direct_prediction", output),
                ("consumer_prediction", actual),
            ):
                drift = float(np.abs(prediction - saved["prediction"]).max())
                maxima[label] = max(maxima[label], drift)
                assert drift <= 2e-8
            summaries(rec, output, y, p, c, None)
            for k, v in curves[name].items():
                compare(rec[k], v)
            checks["selected_models"] += 1
            checks["selected_rows"] += len(x) * 33
            checks["transform_cases"] += 33
            checks["dose_cases"] += 4
        print(
            f"AUDIT {role}:79 curves/25 transformed consumers checked; independent retraining starts",
            flush=True,
        )
        weights = balanced(data["patient"][rows], data["site"][rows])
        for number, trace in enumerate(final_rec["trajectories"], 1):
            method, group, seed, parameter = (
                trace[k] for k in ("method", "group", "seed", "parameter")
            )
            rebuilt, reference_trace = train_reference(
                data["color"][rows], data["target"][rows], weights, group, method, parameter, seed
            )
            maximum_parameter, maximum_output = 0.0, 0.0
            for epoch, diagnostic, primary_diagnostic in zip(
                EPOCHS, reference_trace, trace["checkpoints"], strict=True
            ):
                name = ident(method, group, seed, parameter, epoch)
                expected = final_models[name]
                assert set(rebuilt[epoch]) == set(expected)
                for key in expected:
                    drift = float(
                        np.max(
                            np.abs(rebuilt[epoch][key].astype(float) - expected[key].astype(float))
                        )
                    )
                    maximum_parameter = max(maximum_parameter, drift)
                    np.testing.assert_allclose(
                        rebuilt[epoch][key], expected[key], atol=2e-6, rtol=2e-6
                    )
                for key in (
                    "order_sha256",
                    "examples_seen",
                    "floored_variance_updates",
                    "negative_variance_updates",
                ):
                    assert diagnostic[key] == primary_diagnostic[key], (
                        role,
                        method,
                        group,
                        seed,
                        epoch,
                        key,
                    )
                for key in ("minimum_pre_floor_variance", "minimum_variance"):
                    close(diagnostic[key], primary_diagnostic[key], 1e-8)
                saved = nz(run / "curves" / role / f"{name}.npz")["prediction"]
                drift = float(np.abs(ref_predict(rebuilt[epoch], x) - saved).max())
                maximum_output = max(maximum_output, drift)
                assert drift <= 0.001
                checks["scalar_refit_checkpoints"] += 1
                checks["scalar_refit_clean_rows"] += len(x)
                if name in selected:
                    saved = nz(run / "evaluated" / role / f"{name}.npz")["prediction"]
                    output = np.stack([ref_predict(rebuilt[epoch], z) for z in xx])
                    drift = float(np.abs(output - saved).max())
                    maximum_output = max(maximum_output, drift)
                    assert drift <= 0.001
                    checks["scalar_refit_stress_rows"] += len(x) * 33
            maxima["refit_parameter_absolute"] = max(
                maxima["refit_parameter_absolute"], maximum_parameter
            )
            maxima["refit_prediction"] = max(maxima["refit_prediction"], maximum_output)
            checks["scalar_refit_trajectories"] += 1
            refits.append(
                dict(
                    role=role,
                    method=method,
                    group=group,
                    seed=seed,
                    parameter=parameter,
                    checkpoints=4,
                    max_parameter_absolute=maximum_parameter,
                    max_lab=maximum_output,
                )
            )
            print(
                f"AUDIT refit {role}:{number}/18 {method}/{group}/s{seed} through64 epochs",
                flush=True,
            )
    expected = dict(
        banks=12,
        stored_models=2244,
        new_checkpoints=2160,
        exact_FG_models=84,
        trajectories=540,
        trajectory_checkpoints=2160,
        example_updates=13708800,
        oof_rows=379100,
        candidate_scores=216,
        choices=18,
        clean_models=237,
        clean_rows=94642,
        selected_models=75,
        selected_rows=988350,
        transform_cases=2475,
        dose_cases=300,
        scalar_refit_trajectories=54,
        scalar_refit_checkpoints=216,
        scalar_refit_clean_rows=86256,
        scalar_refit_stress_rows=711612,
    )
    assert checks == expected, (checks, expected)
    paired = []
    rng = np.random.default_rng(771031)
    for role in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for group in GROUPS:
            for method in METHODS:
                candidates = ["fg_norm_static"] + (["adam"] if method != "adam" else [])
                for control in candidates:
                    error = np.mean(
                        [person_errors[(role, method, group, s)] for s in SEEDS], axis=0
                    )
                    baseline = np.mean(
                        [person_errors[(role, control, group, s)] for s in SEEDS], axis=0
                    )
                    diff = error - baseline
                    draws = np.mean(diff[rng.integers(0, len(diff), (20000, len(diff)))], axis=1)
                    paired.append(
                        dict(
                            role=role,
                            group=group,
                            method=method,
                            control=control,
                            people=len(diff),
                            improved_people=int(np.sum(diff < 0)),
                            mean_difference=float(diff.mean()),
                            fixed_prediction_person_bootstrap_95=np.quantile(
                                draws, [0.025, 0.975]
                            ).tolist(),
                        )
                    )
    assert len(paired) == 30
    deps = (
        "scripts/chromaseed_gaussian_reference.py",
        "scripts/chromaseed_gaussian_numpy.py",
        "scripts/chromaseed_feature_groups_audit.py",
        "scripts/chromaseed_feature_groups_numpy.py",
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_gate_stability_audit.py",
        "scripts/chromaseed_gated_audit.py",
        "scripts/chromaseed_kernel_audit.py",
        "scripts/chromaseed_perceptual_audit.py",
        "scripts/chromaseed_refine_audit.py",
        "scripts/skin_local_search_train.py",
        "src/luma_skin_vision/color.py",
    )
    write_json(
        out / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=SOURCE,
            selection_sha256=sha(run / "selections.json"),
            results_sha256=sha(run / "results.json"),
            audit_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in deps},
            checks=checks,
            maxima=maxima,
            scalar_refits=refits,
            variance_diagnostics=variance_diagnostics,
            paired=paired,
            bootstrap_scope="Descriptive fixed-prediction resampling of reused people, not simultaneous or independent confirmation.",
            wall_seconds=time.perf_counter() - started,
        ),
    )
    print(f"TG AUDIT PASS:{checks}; maxima:{maxima}", flush=True)


if __name__ == "__main__":
    main()
