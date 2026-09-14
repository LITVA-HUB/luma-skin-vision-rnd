"""Independent NR representation retraining, all-head QR reconstruction and output audit."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact, summaries
from chromaseed_feature_groups_audit import compare
from chromaseed_gate_stability_audit import close, transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer, direct, ident, row_hash, scoring
from chromaseed_gaussian_reference import initial, normalization, train_reference
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_readout_reference import refit
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
TG = ROOT / "experiments/runs/chromaseed_gaussian_v1"
SOURCE = "70246d9935e52d5be0dcebf758ba3e63b30be4965a9154a19ba180f5cf1782c1"
SEEDS = (17, 29, 43)
EPOCHS = (1, 4, 16)
GROUPS = ("raw36", "mean3")
FAMILIES = ("norm", "perceptual")
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)
ALPHAS = (0.1, 1.0, 10.0)


def name_for(family, basis, group, seed, ai=None):
    name = f"{family}_{basis}_{group}_s{seed}"
    return name if ai is None else name + f"_a{ai}"


def expected_meta():
    expected = {}
    for group in GROUPS:
        for basis in BASES:
            for seed in SEEDS:
                base_name = f"{basis}_{group}_s{seed}"
                for family in FAMILIES:
                    for ai, alpha in enumerate(ALPHAS):
                        expected[name_for(family, basis, group, seed, ai)] = dict(
                            family=family,
                            basis=basis,
                            group=group,
                            seed=seed,
                            alpha=alpha,
                            alpha_index=ai,
                            base_name=base_name,
                            origin="fit",
                        )
                if basis != "random":
                    expected[name_for("unchanged", basis, group, seed)] = dict(
                        family="unchanged",
                        basis=basis,
                        group=group,
                        seed=seed,
                        alpha=None,
                        alpha_index=None,
                        base_name=base_name,
                        origin="fit",
                    )
        for seed in SEEDS:
            name = f"fg_norm_static_{group}_s{seed}"
            expected[name] = dict(
                family="fg_norm_static",
                basis="reference",
                group=group,
                seed=seed,
                alpha=0.1,
                alpha_index=None,
                base_name=None,
                origin="FG",
                parent_model_id=name,
            )
    expected["constant"] = dict(
        family="constant",
        basis="reference",
        group="constant",
        seed=None,
        alpha=None,
        alpha_index=None,
        base_name=None,
        origin="FG",
        parent_model_id="constant",
    )
    return expected


def parameter_check(a, b, maxima):
    assert set(a) == set(b)
    same = True
    for k in a:
        assert a[k].dtype == b[k].dtype
        drift = float(np.max(abs(a[k].astype(float) - b[k].astype(float))))
        maxima["representation_parameter"] = max(maxima["representation_parameter"], drift)
        np.testing.assert_allclose(a[k], b[k], atol=2e-6, rtol=2e-6)
        same &= np.array_equal(a[k], b[k])
    return same


def bank_check(run, data, role, fit, query, fold, checks, maxima, diagnostics, nonexact):
    stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
    path, parent_path = run / stage / role / sub, TG / stage / role / sub
    rec, parent = js(path / "receipt.json"), js(parent_path / "receipt.json")
    assert rec["source_lock_sha256"] == SOURCE
    assert rec["selection_sha256"] == (sha(run / "selections.json") if fold is None else None)
    assert rec["fit_rows_sha256"] == parent["fit_rows_sha256"] == row_hash(fit)
    assert rec["query_rows_sha256"] == (None if fold is None else row_hash(query))
    assert rec["n_fit_rows"] == len(fit) and rec["n_fit_people"] == len(
        np.unique(data["patient"][fit])
    )
    assert not set(data["patient"][fit]) & set(data["patient"][query])
    for directory, receipt in ((path, rec), (parent_path, parent)):
        for file, digest in receipt["files"].items():
            assert sha(directory / file) == digest
    expected = expected_meta()
    assert set(rec["models"]) == set(expected) and len(expected) == 295
    assert len(rec["bases"]) == 42 and len(rec["trajectories"]) == 12
    assert rec["head_solves"] == 252 and rec["imported_FG_models"] == 7
    arrays, base_arrays, parent_arrays = (
        nz(path / "models.npz"),
        nz(path / "bases.npz"),
        nz(parent_path / "models.npz"),
    )
    x, y = data["color"][fit], data["target"][fit]
    weights = balanced(data["patient"][fit], data["site"][fit])
    reconstructed_bases, original_bases = {}, {}
    exact_parent = 0
    for group in GROUPS:
        prep, _, _ = normalization(x, y, group)
        for seed in SEEDS:
            means, _ = initial(36 if group == "raw36" else 3, seed)
            reconstructed_bases[f"random_{group}_s{seed}"] = {
                **prep,
                **{k: v.astype(np.float32) for k, v in means.items()},
            }
            checks["random_bases"] += 1
        for method, parameter in (("adam", 0.001), ("tagi_full3", 1.0)):
            for seed in SEEDS:
                matches = [
                    t
                    for t in rec["trajectories"]
                    if (t["group"], t["method"], t["seed"]) == (group, method, seed)
                ]
                assert len(matches) == 1
                trace = matches[0]
                assert trace["parameter"] == parameter and trace["n_fit_rows"] == len(x)
                parameters = 2563 if group == "raw36" else 451
                assert trace["parameter_count"] == parameters and trace[
                    "training_parameter_state_bytes"
                ] == parameters * (24 if method == "adam" else 16)
                models, rebuilt_trace = train_reference(
                    x, y, weights, group, method, parameter, seed, checkpoints=EPOCHS
                )
                for epoch, rc, pc in zip(EPOCHS, rebuilt_trace, trace["checkpoints"], strict=True):
                    name = f"{method}_e{epoch}_{group}_s{seed}"
                    reconstructed_bases[name] = models[epoch]
                    for key in (
                        "epoch",
                        "examples_seen",
                        "order_sha256",
                        "floored_variance_updates",
                        "negative_variance_updates",
                    ):
                        assert rc[key] == pc[key], (role, fold, name, key)
                    assert (
                        pc["examples_seen"] == epoch * len(x)
                        and pc["negative_variance_updates"] == 0
                    )
                    for key in ("minimum_pre_floor_variance", "minimum_variance"):
                        close(rc[key], pc[key], 1e-8)
                    assert pc["block_elapsed_seconds"] > 0
                    checks["learned_checkpoints"] += 1
                diagnostics.append(dict(role=role, stage=stage, fold=fold, **trace))
                checks["representation_trajectories"] += 1
    assert set(reconstructed_bases) == set(rec["bases"])
    for name, rebuilt in reconstructed_bases.items():
        model, meta = model_from(base_arrays, name), rec["bases"][name]
        original_bases[name] = model
        identical = parameter_check(model, rebuilt, maxima)
        checks["exact_reconstructed_bases"] += int(identical)
        if not identical:
            differences = {
                k: float(np.max(abs(model[k].astype(float) - rebuilt[k].astype(float))))
                for k in model
                if not np.array_equal(model[k], rebuilt[k])
            }
            item = dict(role=role, stage=stage, fold=fold, name=name, differences=differences)
            nonexact.append(item)
            print(f"AUDIT within registered tolerance, not bitwise: {item}", flush=True)
        if meta["method"] != "random":
            parent_name = ident(
                meta["method"],
                meta["group"],
                meta["seed"],
                0.001 if meta["method"] == "adam" else 1.0,
                meta["epoch"],
            )
            assert meta["exact_TG"] == (parent_name in parent["models"])
            if meta["exact_TG"]:
                assert meta["parent_model_id"] == parent_name
                exact(model, model_from(parent_arrays, parent_name))
                exact_parent += 1
            t = rec["trajectories"][meta["trajectory_index"]]
            assert all(t[k] == meta[k] for k in ("method", "group", "seed"))
        else:
            assert meta["trajectory_index"] is None and not meta["exact_TG"]
        checks["basis_payloads"] += 1
    assert exact_parent == rec["exact_TG_bases"]
    checks["exact_TG_bases"] += exact_parent
    checks["example_updates"] += rec["examples_seen_all_trajectories"]
    assert rec["examples_seen_all_trajectories"] == 12 * len(x) * 16
    models, references = {}, {}
    both_x = np.concatenate((x, data["color"][query]))
    for name, expected_row in expected.items():
        meta, model = rec["models"][name], model_from(arrays, name)
        for k, v in expected_row.items():
            assert meta[k] == v, (name, k, meta[k], v)
        assert meta["numeric_bytes"] == sum(v.nbytes for v in model.values())
        models[name] = model
        if meta["origin"] == "FG":
            exact(model, model_from(parent_arrays, meta["parent_model_id"]))
            if name == "constant":
                np.testing.assert_array_equal(
                    model["constant_lab"], np.average(y, weights=weights, axis=0).astype(np.float32)
                )
            checks["exact_FG_models"] += 1
        elif meta["family"] == "unchanged":
            exact(model, original_bases[meta["base_name"]])
            references[name] = reconstructed_bases[meta["base_name"]]
            checks["unchanged_models"] += 1
        else:
            base = original_bases[meta["base_name"]]
            assert meta["numeric_bytes"] == (10564 if meta["group"] == "raw36" else 1855)
            for k in base:
                if k not in ("w2", "b2"):
                    np.testing.assert_array_equal(model[k], base[k])
            independent, info = refit(
                reconstructed_bases[meta["base_name"]], x, y, weights, meta["family"], meta["alpha"]
            )
            assert info["rank"] == (195 if meta["family"] == "perceptual" else 65)
            drift = float(np.abs(direct(model, both_x) - direct(independent, both_x)).max())
            maxima["qr_prediction"] = max(maxima["qr_prediction"], drift)
            assert drift <= 0.001, (role, fold, name, drift)
            fit_info = meta["fit_diagnostics"]
            assert 0 <= fit_info["normal_residual"] < 1e-8 and fit_info["fit_seconds"] > 0
            assert fit_info["max_folded_lab_drift"] <= 0.001
            maxima["normal_residual"] = max(maxima["normal_residual"], fit_info["normal_residual"])
            maxima["folded_prediction"] = max(
                maxima["folded_prediction"], fit_info["max_folded_lab_drift"]
            )
            if info["metric_scale"] is not None:
                close(info["metric_scale"], fit_info["metric_scale"], 1e-4)
            references[name] = independent
            checks["qr_head_refits"] += 1
            checks["qr_fit_rows"] += len(x)
            checks["qr_query_rows"] += len(query)
        checks["stored_models"] += 1
    assert set(arrays) == {name + "__" + k for name, m in models.items() for k in m}
    assert set(base_arrays) == {name + "__" + k for name, m in original_bases.items() for k in m}
    if fold is not None:
        oof, old_oof = nz(path / "oof.npz"), nz(parent_path / "oof.npz")
        assert set(oof) == {"row_indices"} | {"pred__" + name for name in models}
        np.testing.assert_array_equal(oof["row_indices"], query)
        for name, model in models.items():
            actual = direct(model, data["color"][query])
            saved = oof["pred__" + name]
            assert saved.shape == (len(query), 3)
            drift = float(np.abs(actual - saved).max())
            maxima["oof_prediction"] = max(maxima["oof_prediction"], drift)
            assert drift <= 2e-8
            m = rec["models"][name]
            if m["origin"] == "FG":
                np.testing.assert_array_equal(saved, old_oof["pred__" + name])
            elif m["family"] == "unchanged":
                bm = rec["bases"][m["base_name"]]
                if bm["exact_TG"]:
                    np.testing.assert_array_equal(saved, old_oof["pred__" + bm["parent_model_id"]])
            checks["oof_rows"] += len(query)
    checks["banks"] += 1
    print(
        f"AUDIT {stage}/{role}/{sub}:12 independent trajectories,252 QR heads,all payloads/outputs checked",
        flush=True,
    )
    return models, references, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    assert not (out / "verification.json").exists(), "sealed audit is read-only"
    start = time.perf_counter()
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    lock, selection, result = (
        js(run / p) for p in ("source_lock.json", "selections.json", "results.json")
    )
    assert (
        sha(run / "source_lock.json")
        == selection["source_lock_sha256"]
        == result["source_lock_sha256"]
        == SOURCE
    )
    assert result["selection_sha256"] == sha(run / "selections.json")
    assert len(lock["sources"]) == 84 and len(lock["input_sha256"]) == 44
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("color", "target", "patient", "site", "device")}
    checks, maxima, diagnostics, errors, nonexact = Counter(), Counter(), [], {}, []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        fit, held = np.flatnonzero(fit), np.flatnonzero(held)
        folds = folds_for(data["patient"][fit], data["device"][fit])
        merged = {n: np.empty((len(fit), 3)) for n in expected_meta()}
        seen = []
        for fold in range(3):
            rows, query = fit[folds != fold], fit[folds == fold]
            bank_check(run, data, role, rows, query, fold, checks, maxima, diagnostics, nonexact)
            saved = nz(run / "inner" / role / f"fold{fold}" / "oof.npz")
            index = np.searchsorted(fit, query)
            seen.extend(index.tolist())
            for name in merged:
                merged[name][index] = saved["pred__" + name]
        np.testing.assert_array_equal(sorted(seen), np.arange(len(fit)))
        for group in GROUPS:
            for family in FAMILIES:
                entry, winners = selection["roles"][role][group][family], []
                assert set(entry["bases"]) == set(BASES)
                for basis in BASES:
                    candidates = []
                    for ai, alpha in enumerate(ALPHAS):
                        scores = [
                            scoring(
                                merged[name_for(family, basis, group, seed, ai)],
                                data["target"][fit],
                                data["patient"][fit],
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
                    compare(entry["bases"][basis]["candidates"], candidates)
                    winner = sorted(
                        candidates, key=lambda c: (c["clean"], c["p90"], c["alpha_index"])
                    )[0]
                    compare(entry["bases"][basis]["selected"], winner)
                    winners.append(winner)
                    checks["candidate_scores"] += 3
                    checks["choices"] += 1
                policy = sorted(
                    winners,
                    key=lambda c: (
                        c["clean"],
                        c["p90"],
                        0 if c["basis"] == "random" else int(c["basis"].rsplit("_e", 1)[1]),
                        BASES.index(c["basis"]),
                    ),
                )[0]
                compare(entry["policy"], policy)
                checks["policies"] += 1
        refs = {
            n: scoring(p, data["target"][fit], data["patient"][fit])
            for n, p in merged.items()
            if n.startswith(("unchanged", "fg_")) or n == "constant"
        }
        compare(selection["references"][role], refs)
        models, qr_models, bank = bank_check(
            run, data, role, fit, held, None, checks, maxima, diagnostics, nonexact
        )
        records = {r["name"]: r for r in result["records"] if r["role"] == role}
        expected_selected = set()
        for name, meta in bank["models"].items():
            if (
                meta["family"] not in FAMILIES
                or meta["alpha_index"]
                == selection["roles"][role][meta["group"]][meta["family"]]["bases"][meta["basis"]][
                    "selected"
                ]["alpha_index"]
            ):
                expected_selected.add(name)
        assert set(records) == expected_selected and len(records) == 127
        x, y, person, site, camera = (
            data[k][held] for k in ("color", "target", "patient", "site", "device")
        )
        xx = [transformed(x, d, anchor) for d, anchor in SETTINGS]
        for name, rec in records.items():
            mp, pp = (
                run / "selected" / role / f"{name}.npz",
                run / "evaluated" / role / f"{name}.npz",
            )
            assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
            model, saved = nz(mp), nz(pp)
            exact(model, models[name])
            for k, v in bank["models"][name].items():
                compare(rec[k], v)
            is_head = rec["family"] in FAMILIES
            assert rec["policy_selected"] == (
                is_head
                and rec["basis"]
                == selection["roles"][role][rec["group"]][rec["family"]]["policy"]["basis"]
            )
            assert rec["archive_bytes"] == mp.stat().st_size
            np.testing.assert_array_equal(saved["row_indices"], held)
            assert saved["prediction"].shape == (33, len(x), 3)
            direct_output = np.stack([direct(model, z) for z in xx])
            consumer = actual_consumer(model)
            actual = np.array([[consumer(row) for row in z] for z in xx])
            for label, output in (
                ("direct_prediction", direct_output),
                ("consumer_prediction", actual),
            ):
                drift = float(np.abs(output - saved["prediction"]).max())
                maxima[label] = max(maxima[label], drift)
                assert drift <= 2e-8
            if name in qr_models:
                output = np.stack([direct(qr_models[name], z) for z in xx])
                drift = float(np.abs(output - saved["prediction"]).max())
                maxima["qr_stress_prediction"] = max(maxima["qr_stress_prediction"], drift)
                assert drift <= 0.001, (role, name, drift)
                checks["refit_stress_rows"] += 33 * len(x)
            clean, per_person = error_summary(direct_output[0], y, person, site)
            compare(rec["metrics"], clean)
            summaries(rec, direct_output, y, person, camera, None)
            errors[(role, rec["family"], rec["basis"], rec["group"], rec["seed"])] = per_person
            checks["selected_models"] += 1
            checks["final_rows"] += len(x) * 33
            checks["transform_cases"] += 33
            checks["dose_cases"] += 4
        print(
            f"AUDIT {role}:84 choices across roles in progress;127 actual consumers x33 transforms checked",
            flush=True,
        )
    expected = dict(
        banks=12,
        representation_trajectories=144,
        learned_checkpoints=432,
        random_bases=72,
        basis_payloads=504,
        exact_reconstructed_bases=504 - len(nonexact),
        exact_TG_bases=369,
        example_updates=979200,
        qr_head_refits=3024,
        qr_fit_rows=1285200,
        qr_query_rows=730296,
        stored_models=3540,
        unchanged_models=432,
        exact_FG_models=84,
        oof_rows=501500,
        candidate_scores=252,
        choices=84,
        policies=12,
        selected_models=381,
        final_rows=5020818,
        transform_cases=12573,
        dose_cases=1524,
        refit_stress_rows=4744080,
    )
    assert dict(checks) == expected, (dict(checks), expected)
    assert 0 <= checks["exact_reconstructed_bases"] <= checks["basis_payloads"]
    rng, paired = np.random.default_rng(771031), []
    for role in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for group in GROUPS:
            for family in FAMILIES:
                for basis in BASES:
                    pairs = [("fg_norm_static", "reference", "head_vs_FG")]
                    if basis != "random":
                        pairs.append(("unchanged", basis, "head_vs_unchanged"))
                    if basis == selection["roles"][role][group][family]["policy"]["basis"]:
                        pairs.append(("fg_norm_static", "reference", "policy_vs_FG"))
                    for control, control_basis, kind in pairs:
                        current = np.mean(
                            [errors[(role, family, basis, group, s)] for s in SEEDS], axis=0
                        )
                        reference = np.mean(
                            [errors[(role, control, control_basis, group, s)] for s in SEEDS],
                            axis=0,
                        )
                        diff = current - reference
                        draws = np.mean(
                            diff[rng.integers(0, len(diff), (20000, len(diff)))], axis=1
                        )
                        paired.append(
                            dict(
                                role=role,
                                group=group,
                                family=family,
                                basis=basis,
                                control=control,
                                kind=kind,
                                people=len(diff),
                                improved_people=int(np.sum(diff < 0)),
                                mean_difference=float(diff.mean()),
                                fixed_prediction_person_bootstrap_95=np.quantile(
                                    draws, [0.025, 0.975]
                                ).tolist(),
                            )
                        )
    assert len(paired) == 168
    deps = (
        "scripts/chromaseed_neural_readout_reference.py",
        "scripts/chromaseed_gaussian_reference.py",
        "scripts/chromaseed_gaussian_audit.py",
        "scripts/chromaseed_gaussian_numpy.py",
        "scripts/chromaseed_perceptual_reference.py",
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_feature_groups_audit.py",
        "scripts/chromaseed_feature_groups_numpy.py",
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
            checks=dict(checks),
            maxima=dict(maxima),
            variance_diagnostics=diagnostics,
            non_bitwise_representations=nonexact,
            paired=paired,
            bootstrap_scope="20000 fixed-prediction person draws; descriptive on reused/confounded roles, no retraining or selection uncertainty",
            wall_seconds=time.perf_counter() - start,
        ),
    )
    print(
        "NR AUDIT PASS:all144 representation refits,3024 QR heads and381 consumers verified",
        flush=True,
    )


if __name__ == "__main__":
    main()
