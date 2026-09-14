"""Independent FG payload, geometry, person selection, consumer and QR audit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact, summaries, validate_normalizer
from chromaseed_feature_groups_numpy import Predictor
from chromaseed_gate_stability_audit import close, transformed
from chromaseed_gated_audit import model_from, reference_gate
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_projection_reference import direct_width, latent, refit
from chromaseed_projection_reference import predict as reference_predict
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "f6a8ac340650ae5d62004f59916ba720b702e6dc38f1fedf6928f5797b53f4cf"
GROUPS = {
    "raw36": list(range(36)),
    "mean3": [27, 28, 29],
    "median3": [12, 13, 14],
    "central9": list(range(9, 18)),
    "mean_std6": list(range(27, 33)),
    "quant27": list(range(27)),
    "no_corr33": list(range(33)),
}
ALL_GROUPS = (*GROUPS, "projected16")
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
SEEDS, ALPHAS = (17, 29, 43), (0.1, 1.0, 10.0)
SETTINGS = [(0.0, np.zeros(3))] + [
    (v / 255, np.array(a)) for v in (1, 4, 16, 64) for a in itertools.product((0.0, 1.0), repeat=3)
]
ERRATUM = "docs/research/chromaseed_feature_groups_count_erratum.md"


def ident(family, seed, group, alpha):
    return f"{family}_{group}_s{seed}_a{ALPHAS.index(alpha)}"


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def subset(x, group):
    return x if group == "projected16" else x[:, GROUPS[group]]


def direct(model, x):
    clean = {k: v for k, v in model.items() if k != "feature_indices"}
    xx = x if "feature_indices" not in model else x[:, model["feature_indices"]]
    return reference_predict(clean, xx)


def gate(model, x):
    xx = x if "feature_indices" not in model else x[:, model["feature_indices"]]
    z = norm(model, xx)
    return np.sum(z * model["gate_beta"][None, 1:].astype(np.float64), axis=1) + float(
        model["gate_beta"][0]
    )


def cached_output(model, kernel, x):
    if "constant_lab" in model:
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    value = kernel @ model["coefficient"].astype(np.float64)
    if "correction" in model:
        value += (
            float(model["rho"])
            * np.clip(gate(model, x), -1, 1)[:, None]
            * (kernel @ model["correction"].astype(np.float64))
        )
    return value * model["y_std"] + model["y_mean"]


def scoring(pred, y, p):
    e = delta_e00(pred, y)
    return dict(
        clean=float(sum(e[p == k].mean() for k in np.unique(p)) / len(np.unique(p))),
        p90=float(np.percentile(e, 90)),
    )


def compare(a, b):
    if isinstance(b, dict):
        assert set(a) == set(b)
        for k in b:
            compare(a[k], b[k])
    elif isinstance(b, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            compare(x, y)
    elif isinstance(b, str) or b is None:
        assert a == b
    else:
        close(a, b)


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    start, run, out = time.perf_counter(), args.run, args.output
    lock, result, selected = (
        js(run / p) for p in ("source_lock.json", "results.json", "selections.json")
    )
    assert (
        sha(run / "source_lock.json")
        == SOURCE
        == result["source_lock_sha256"]
        == selected["source_lock_sha256"]
    )
    assert result["selection_sha256"] == sha(run / "selections.json")
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH == lock["cache_sha256"]
    assert lock["groups"] == GROUPS and lock["all_groups"] == list(ALL_GROUPS)
    assert (
        lock["families"] == list(FAMILIES)
        and lock["seeds"] == list(SEEDS)
        and lock["alphas"] == list(ALPHAS)
    )
    assert lock["rank"] == 128 and lock["clean_allowance"] == 0.05 and lock["p90_allowance"] == 0.1
    assert len(lock["sources"]) == 73 and len(lock["input_sha256"]) == 82
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    for spec, (dose, anchor) in zip(lock["final_settings"], SETTINGS, strict=True):
        close(spec["dose"], dose)
        np.testing.assert_array_equal(spec["anchor"], anchor)
    assert (ROOT / ERRATUM).is_file()
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    checks = {
        k: 0
        for k in (
            "banks",
            "stored_readouts",
            "exact_A_payloads",
            "imported_X_payloads",
            "constant_fits",
            "aliases",
            "width_checks",
            "dense_pivot_paths",
            "pivot_path_differences",
            "rank_differences",
            "gate_reference_checks",
            "oof_query_rows",
            "candidate_scores",
            "group_choices",
            "policy_choices",
            "selected_models",
            "final_query_rows",
            "final_transform_cases",
            "dose_cases",
            "qr_refits",
            "qr_query_rows",
            "new_coefficient_solutions",
            "reduced_coefficient_solutions",
            "gram_decompositions",
            "basis_preparations",
            "exact_width_calculations",
            "gate_fits",
            "auxiliary_baseline_solves",
            "auxiliary_theta_solves",
            "target_metric_preparations",
        )
    }
    maxima = {
        k: 0.0
        for k in (
            "oof_prediction",
            "direct_prediction",
            "consumer_prediction",
            "qr_prediction",
            "gate_parameter",
            "normal_residual",
        )
    }
    references, geometry, person_errors = [], [], {}
    for role, (fit, held) in masks.items():
        rows = np.flatnonzero(fit)
        folds = folds_for(data["patient"][rows], data["device"][rows])
        merged, sizes, seen = {}, {}, []
        for fold in (*range(3), None):
            fr = rows if fold is None else rows[folds != fold]
            qr = None if fold is None else rows[folds == fold]
            stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
            path = run / stage / role / sub
            rec = js(path / "receipt.json")
            assert rec["source_lock_sha256"] == SOURCE and rec["fit_rows_sha256"] == row_hash(fr)
            assert rec["query_rows_sha256"] == (None if qr is None else row_hash(qr))
            assert rec["n_fit_rows"] == len(fr) and rec["n_fit_people"] == len(
                np.unique(data["patient"][fr])
            )
            if qr is not None:
                assert not set(data["patient"][fr]) & set(data["patient"][qr])
            for p, h in rec["files"].items():
                assert sha(path / p) == h
            arrays = nz(path / "models.npz")
            models = {name: model_from(arrays, name) for name in rec["models"]}
            assert len(models) == 289 and len(arrays) == sum(len(m) for m in models.values())
            old = {}
            for label, parent in (("A", "affine"), ("X", "projection")):
                pp = ROOT / f"experiments/runs/chromaseed_{parent}_v1" / stage / role / sub
                rr = js(pp / "receipt.json")
                assert rr["fit_rows_sha256"] == row_hash(fr)
                old[label] = nz(pp / "models.npz")
            x, y, p, s, c = (data[k][fr] for k in ("color", "target", "patient", "site", "device"))
            w, two = balanced(p, s), len(np.unique(c)) > 1
            close(rec["original_weight_mass"], w.sum(), 2e-10)
            assert rec["new_coefficient_solutions"] == (252 if two else 126)
            assert rec["gram_decompositions"] == (42 if two else 21)
            assert rec["basis_preparations"] == 21 and rec["exact_width_calculations"] == 7
            assert rec["gate_fits"] == 7 * int(two) and rec["target_metric_preparations"] == 1
            for k in (
                "new_coefficient_solutions",
                "gram_decompositions",
                "basis_preparations",
                "exact_width_calculations",
                "gate_fits",
                "auxiliary_baseline_solves",
                "auxiliary_theta_solves",
                "target_metric_preparations",
            ):
                checks[k] += rec[k]
            for op in rec["operations"]:
                assert len(op["solutions"]) == 6
                if op["group"] != "raw36":
                    checks["reduced_coefficient_solutions"] += 6
                for sol in op["solutions"]:
                    assert np.isfinite(
                        [sol["objective"], sol["objective_minus_zero"], sol["normal_residual"]]
                    ).all()
                    assert sol["normal_residual"] < 1e-8 and sol["objective_minus_zero"] <= 1e-8
                    maxima["normal_residual"] = max(
                        maxima["normal_residual"], sol["normal_residual"]
                    )
            info = {(v["group"], v["seed"]): v for v in rec["bases"]}
            kernels = {}
            for group in ALL_GROUPS:
                xm = subset(x, group)
                m = models[ident("norm_static", 17, group, 0.1)]
                validate_normalizer(m, xm, y)
                q = latent(m, xm)
                width = float(np.float32(direct_width(q)))
                assert width == float(m["width"])
                checks["width_checks"] += 1
                dense = direct_kernel(q, q, width)
                if group != "projected16":
                    state_info = next(v for v in rec["groups"] if v["group"] == group)
                    assert state_info["dimension"] == len(GROUPS[group])
                    close(state_info["base_width"], direct_width(q), 2e-10)
                    if two:
                        expected_beta = reference_gate(q, p, s, c)
                        actual_beta = models[ident("norm_joint_soft", 17, group, 0.1)]["gate_beta"]
                        delta = float(np.max(np.abs(expected_beta - actual_beta)))
                        maxima["gate_parameter"] = max(maxima["gate_parameter"], delta)
                        assert delta <= 1e-5
                        checks["gate_reference_checks"] += 1
                for seed in SEEDS:
                    mm = models[ident("norm_static", seed, group, 0.1)]
                    ids, _, _ = select_landmarks(dense, w, "rpchol", 128, seed)
                    if group != "projected16":
                        bi = info[(group, seed)]
                        actual_ids = np.asarray(bi["fit_center_indices"], int)
                        np.testing.assert_array_equal(
                            mm["centers"], q[actual_ids].astype(np.float32)
                        )
                        assert mm["centers"].shape == (bi["actual_centers"], len(GROUPS[group]))
                        same_path = np.array_equal(ids, actual_ids)
                        checks["pivot_path_differences"] += int(not same_path)
                        ev = np.linalg.eigvalsh(dense[np.ix_(actual_ids, actual_ids)])
                        erank = int(np.sum(ev > 1e-8 * ev.max()))
                        rank_same = erank == bi["effective_rank"]
                        checks["rank_differences"] += int(not rank_same)
                        geometry.append(
                            dict(
                                role=role,
                                fold=fold,
                                group=group,
                                seed=seed,
                                same_pivot_path=same_path,
                                primary_rank=bi["effective_rank"],
                                independent_rank=erank,
                                actual_centers=len(actual_ids),
                            )
                        )
                    else:
                        np.testing.assert_array_equal(mm["centers"], q[ids].astype(np.float32))
                    checks["dense_pivot_paths"] += 1
                    if qr is not None:
                        kernels[(group, seed)] = direct_kernel(
                            latent(mm, subset(data["color"][qr], group)), mm["centers"], width
                        )
            aliases = 0
            for name, m in models.items():
                meta = rec["models"][name]
                Predictor(m)
                assert meta["numeric_bytes"] == sum(v.nbytes for v in m.values())
                if name == "constant":
                    exact(m, dict(constant_lab=np.average(y, axis=0, weights=w).astype(np.float32)))
                    checks["constant_fits"] += 1
                    continue
                family, group, seed, alpha = (meta[k] for k in ("family", "group", "seed", "alpha"))
                assert name == ident(family, seed, group, alpha)
                if group == "raw36":
                    prior = f"{family}_s{seed}_a{ALPHAS.index(alpha)}_e0"
                    exact(m, model_from(old["A"], prior))
                    checks["exact_A_payloads"] += 1
                elif group == "projected16":
                    prior = f"{family}_d16_t05_s{seed}_a{ALPHAS.index(alpha)}"
                    exact(m, model_from(old["X"], prior))
                    checks["imported_X_payloads"] += 1
                else:
                    np.testing.assert_array_equal(
                        m["feature_indices"], np.array(GROUPS[group], np.uint8)
                    )
                    validate_normalizer(m, subset(x, group), y)
                active = two and family.endswith("joint_soft")
                assert ("gate_beta" in m) == active
                template = models[ident("norm_static", seed, group, 0.1)]
                for key in ("x_mean", "x_std", "y_mean", "y_std", "centers", "width"):
                    np.testing.assert_array_equal(m[key], template[key])
                if active:
                    assert float(m["rho"]) == 1 and int(m["gate_mode"]) == 1
                if meta["fallback"]:
                    assert group != "projected16" and not two
                    exact(
                        m, models[ident(family.replace("joint_soft", "static"), seed, group, alpha)]
                    )
                    aliases += 1
            assert aliases == rec["exact_joint_static_aliases"] == (0 if two else 126)
            assert (
                rec["exact_A_payloads"] == 36
                and rec["imported_X_payloads"] == 36
                and rec["constant_fits"] == 1
            )
            checks["aliases"] += aliases
            checks["banks"] += 1
            checks["stored_readouts"] += len(models)
            if qr is not None:
                oof = nz(path / "oof.npz")
                np.testing.assert_array_equal(oof["row_indices"], qr)
                position = np.searchsorted(rows, qr)
                seen.extend(position.tolist())
                for name, m in models.items():
                    meta = rec["models"][name]
                    kernel = None if name == "constant" else kernels[(meta["group"], meta["seed"])]
                    pred = cached_output(m, kernel, data["color"][qr])
                    drift = float(np.max(np.abs(pred - oof["pred__" + name])))
                    maxima["oof_prediction"] = max(maxima["oof_prediction"], drift)
                    assert drift <= 2e-8
                    if name not in merged:
                        merged[name] = np.empty((len(rows), 3))
                    merged[name][position] = pred
                    sizes[name] = max(sizes.get(name, 0), meta["numeric_bytes"])
                    checks["oof_query_rows"] += len(qr)
            print(
                f"AUDIT {stage}/{role}/{sub}:289 payloads,8 widths,24 dense paths checked",
                flush=True,
            )
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))
        yy, pp = data["target"][rows], data["patient"][rows]
        for family in FAMILIES:
            winners = []
            saved = selected["roles"][role][family]
            for order, group in enumerate(ALL_GROUPS):
                candidates = []
                for alpha in ALPHAS:
                    names = [ident(family, seed, group, alpha) for seed in SEEDS]
                    ss = [scoring(merged[name], yy, pp) for name in names]
                    candidates.append(
                        dict(
                            family=family,
                            group=group,
                            order=order,
                            alpha=alpha,
                            clean=float(np.mean([v["clean"] for v in ss])),
                            p90=float(np.mean([v["p90"] for v in ss])),
                            numeric_bytes=max(sizes[n] for n in names),
                            seed_scores=ss,
                        )
                    )
                compare(saved["groups"][group]["candidates"], candidates)
                winner = sorted(candidates, key=lambda r: (r["clean"], r["p90"], -r["alpha"]))[0]
                compare(saved["groups"][group]["selected"], winner)
                winners.append(winner)
                checks["candidate_scores"] += 3
                checks["group_choices"] += 1
            raw = next(v for v in winners if v["group"] == "raw36")
            eligible = [
                v
                for v in winners
                if v["clean"] <= raw["clean"] + 0.05 and v["p90"] <= raw["p90"] + 0.1
            ]
            choices = {
                "quality": sorted(
                    winners, key=lambda r: (r["clean"], r["numeric_bytes"], r["p90"], r["order"])
                )[0],
                "compact": sorted(
                    eligible, key=lambda r: (r["numeric_bytes"], r["clean"], r["p90"], r["order"])
                )[0],
            }
            for policy, choice in choices.items():
                compare(saved["policies"][policy], choice)
                checks["policy_choices"] += 1
        compare(selected["references"][role], scoring(merged["constant"], yy, pp))
        qr = np.flatnonzero(held)
        x, y, p, s, c = (data[k][qr] for k in ("color", "target", "patient", "site", "device"))
        assert not set(p) & set(data["patient"][rows])
        xx = [transformed(x, t, a) for t, a in SETTINGS]
        bank = nz(run / "final" / role / "bank/models.npz")
        records = [r for r in result["records"] if r["role"] == role]
        expected_cases = {(f, g, seed) for f in FAMILIES for g in ALL_GROUPS for seed in SEEDS} | {
            ("constant", "constant", None)
        }
        assert {(r["family"], r["group"], r["seed"]) for r in records} == expected_cases and len(
            records
        ) == 97
        for index, rec in enumerate(records):
            mp = run / "selected" / role / f"{rec['name']}.npz"
            ep = run / "evaluated" / role / f"{rec['name']}.npz"
            assert sha(mp) == rec["model_sha256"] and sha(ep) == rec["prediction_sha256"]
            model, saved = nz(mp), nz(ep)
            exact(model, model_from(bank, rec["model_id"]))
            np.testing.assert_array_equal(saved["row_indices"], qr)
            assert saved["prediction"].shape == (33, len(qr), 3)
            assert (
                rec["numeric_bytes"] == sum(v.nbytes for v in model.values())
                and rec["archive_bytes"] == mp.stat().st_size
            )
            assert rec["active_gate"] == ("gate_beta" in model)
            if rec["family"] != "constant":
                chosen = selected["roles"][role][rec["family"]]["groups"][rec["group"]]["selected"]
                assert rec["alpha"] == chosen["alpha"] and rec["model_id"] == ident(
                    rec["family"], rec["seed"], rec["group"], rec["alpha"]
                )
                assert rec["input_dimensions"] == len(model["x_mean"])
                assert rec["kernel_dimensions"] == model["centers"].shape[1] and rec[
                    "actual_centers"
                ] == len(model["centers"])
            direct_output = np.stack([direct(model, v) for v in xx])
            drift = float(np.max(np.abs(direct_output - saved["prediction"])))
            maxima["direct_prediction"] = max(maxima["direct_prediction"], drift)
            assert drift <= 2e-8
            consumer = Predictor(model)
            output = np.array([[consumer(v) for v in transformed_x] for transformed_x in xx])
            drift = float(np.max(np.abs(output - saved["prediction"])))
            maxima["consumer_prediction"] = max(maxima["consumer_prediction"], drift)
            assert drift <= 2e-8
            gates = None if "gate_beta" not in model else np.stack([gate(model, v) for v in xx])
            if gates is not None:
                np.testing.assert_allclose(gates, saved["scores"], rtol=0, atol=2e-10)
            metric, errors = error_summary(output[0], y, p, s)
            close(rec["metrics"], metric)
            summaries(rec, output, y, p, c, gates)
            person_errors[(role, rec["family"], rec["group"], rec["seed"])] = errors
            if rec["family"] != "constant":
                group = rec["group"]
                reference, info = refit(
                    model,
                    subset(data["color"][rows], group),
                    *(data[k][rows] for k in ("target", "patient", "site", "device")),
                    rec["family"],
                    rec["alpha"],
                    "d16_t05" if group == "projected16" else "raw",
                    rec["seed"],
                )
                reconstructed = np.stack(
                    [reference_predict(reference, subset(v, group)) for v in xx]
                )
                drift = float(np.max(np.abs(reconstructed - saved["prediction"])))
                maxima["qr_prediction"] = max(maxima["qr_prediction"], drift)
                references.append(
                    dict(role=role, name=rec["name"], group=group, max_lab_drift=drift, **info)
                )
                assert drift <= 0.001, (role, rec["name"], drift)
                checks["qr_refits"] += 1
                checks["qr_query_rows"] += 33 * len(qr)
            checks["selected_models"] += 1
            checks["final_query_rows"] += 33 * len(qr)
            checks["final_transform_cases"] += 33
            checks["dose_cases"] += 4
            if index % 16 == 15 or index == 96:
                print(
                    f"AUDIT final {role}:{index + 1}/97 consumers, direct predictions and refits checked",
                    flush=True,
                )
    expected = dict(
        banks=12,
        stored_readouts=3468,
        exact_A_payloads=432,
        imported_X_payloads=432,
        constant_fits=12,
        aliases=1008,
        width_checks=96,
        dense_pivot_paths=288,
        gate_reference_checks=28,
        oof_query_rows=491300,
        candidate_scores=288,
        group_choices=96,
        policy_choices=24,
        selected_models=291,
        final_query_rows=3834798,
        final_transform_cases=9603,
        dose_cases=1164,
        qr_refits=288,
        qr_query_rows=3795264,
        new_coefficient_solutions=2016,
        reduced_coefficient_solutions=1728,
        gram_decompositions=336,
        basis_preparations=252,
        exact_width_calculations=84,
        gate_fits=28,
        auxiliary_baseline_solves=36,
        auxiliary_theta_solves=36,
        target_metric_preparations=12,
    )
    for key, value in expected.items():
        assert checks[key] == value, (key, checks[key], value)
    paired, rng = [], np.random.default_rng(84291)
    for role in masks:
        for family in FAMILIES:
            for group in ALL_GROUPS[1:]:
                differences = np.mean(
                    [
                        person_errors[(role, family, group, seed)]
                        - person_errors[(role, family, "raw36", seed)]
                        for seed in SEEDS
                    ],
                    axis=0,
                )
                samples = rng.integers(0, len(differences), (20000, len(differences)))
                paired.append(
                    dict(
                        role=role,
                        family=family,
                        group=group,
                        kind="historical_X" if group == "projected16" else "new_subset",
                        people=len(differences),
                        improved_people=int(np.sum(differences < 0)),
                        mean_difference=float(differences.mean()),
                        fixed_prediction_person_bootstrap_95=np.quantile(
                            differences[samples].mean(1), [0.025, 0.975]
                        ).tolist(),
                    )
                )
    deps = (
        "scripts/chromaseed_feature_groups_numpy.py",
        "scripts/chromaseed_projection_reference.py",
        "scripts/chromaseed_projection_numpy.py",
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_affine_reference.py",
        "scripts/chromaseed_gated_audit.py",
        "scripts/chromaseed_gate_stability_audit.py",
        "scripts/chromaseed_kernel.py",
        "scripts/chromaseed_kernel_audit.py",
        "scripts/chromaseed_perceptual_audit.py",
        "scripts/chromaseed_perceptual_reference.py",
        "scripts/chromaseed_refine_audit.py",
        "scripts/skin_local_search_train.py",
        "src/luma_skin_vision/color.py",
        ERRATUM,
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
            geometry=geometry,
            independent_refits=references,
            paired=paired,
            protocol_count_erratum=ERRATUM,
            correct_final_row_predictions=3834798,
            bootstrap_scope="Descriptive resampling of fixed predictions on reused people; not independent confirmation or simultaneous inference.",
            wall_seconds=time.perf_counter() - start,
        ),
    )
    print(f"FG AUDIT PASS:{checks}, maxima:{maxima}", flush=True)


if __name__ == "__main__":
    main()
