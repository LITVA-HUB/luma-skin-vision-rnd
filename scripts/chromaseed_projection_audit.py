"""Independent X geometry, controls, selection, portable queries and refits."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact, summaries, validate_normalizer
from chromaseed_gate_stability_audit import ANCHORS, DOSES, close, transformed
from chromaseed_gate_stability_audit import score as gate_score
from chromaseed_gated_audit import model_from
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_projection_numpy import Predictor
from chromaseed_projection_reference import (
    direct_width,
    latent,
    predict,
    projector,
    refit,
    spectrum,
)
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
SEEDS, ALPHAS = (17, 29, 43), (0.1, 1.0, 10.0)
REPS = [("raw", 36, None)] + [
    (f"d{d}_t{label}", d, t)
    for d in (8, 16, 36)
    for label, t in (("0", 0.0), ("01", 0.1), ("05", 0.5), ("1", 1.0))
]
CONTROLS = (
    "g_norm_soft",
    "g_perceptual_soft",
    "a_norm_joint_guarded",
    "a_perceptual_joint_guarded",
)
SETTINGS = [(0.0, np.zeros(3))] + [(t, a) for t in DOSES for a in ANCHORS]


def ident(family, seed, rep, alpha):
    return f"{family}_{rep}_s{seed}_a{ALPHAS.index(alpha)}"


def ahash(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def scoring(pred, y, p):
    error = delta_e00(pred, y)
    return dict(
        clean=float(sum(error[p == v].mean() for v in np.unique(p)) / len(np.unique(p))),
        p90=float(np.percentile(error, 90)),
    )


def cached_output(m, k, x):
    if "constant_lab" in m:
        return np.broadcast_to(m["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    value = k @ m["coefficient"].astype(np.float64)
    if "correction" in m:
        value += (
            float(m["rho"])
            * np.clip(gate_score(m, x), -1, 1)[:, None]
            * (k @ m["correction"].astype(np.float64))
        )
    return value * m["y_std"] + m["y_mean"]


def numeric_equal(a, b):
    assert set(a) == set(b)
    for k, v in b.items():
        if isinstance(v, str):
            assert a[k] == v
        else:
            close(a[k], v)


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    pa, pg = (
        ROOT / "experiments/runs/chromaseed_affine_v1",
        ROOT / "experiments/runs/chromaseed_gated_v1",
    )
    start = time.perf_counter()
    lock, result, selected = (
        js(run / p) for p in ("source_lock.json", "results.json", "selections.json")
    )
    source = sha(run / "source_lock.json")
    assert source == result["source_lock_sha256"] == selected["source_lock_sha256"]
    assert result["selection_sha256"] == sha(run / "selections.json")
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH == lock["cache_sha256"]
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    assert len(lock["sources"]) == 59 and len(lock["input_sha256"]) == 145
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    checks = dict(
        banks=0,
        stored_readouts=0,
        exact_A_raw_payloads=0,
        exact_aliases=0,
        exact_imported_controls=0,
        constant_fits=0,
        covariance_svd_checks=0,
        projection_checks=0,
        width_checks=0,
        dense_landmark_checks=0,
        oof_query_rows=0,
        candidate_score_pairs=0,
        reference_score_groups=0,
        choices=0,
        selected_models=0,
        final_query_rows=0,
        final_transform_cases=0,
        dose_cases=0,
        qr_selected_refits=0,
        qr_positive_probes=0,
        qr_query_rows=0,
        new_coefficient_solutions=0,
        gram_decompositions=0,
        basis_preparations=0,
        exact_width_calculations=0,
        covariance_eigendecompositions=0,
        gate_fits=0,
        auxiliary_baseline_solves=0,
        auxiliary_theta_solves=0,
    )
    maxima = dict(
        projection_parameter=0.0,
        projection_metric_relative=0.0,
        covariance_spectrum=0.0,
        oof_prediction=0.0,
        standalone_prediction=0.0,
        qr_prediction=0.0,
        normal_residual=0.0,
        full_rotation_fit_kernel=0.0,
        width=0.0,
    )
    refs, probes, pe, geometry = [], [], {}, []
    for role, (fit, held) in masks.items():
        rows = np.flatnonzero(fit)
        fold_ids = folds_for(data["patient"][rows], data["device"][rows])
        merged, sizes, seen = {}, {}, []
        for fold in (*range(3), None):
            fr = rows if fold is None else rows[fold_ids != fold]
            qr = None if fold is None else rows[fold_ids == fold]
            stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
            path = run / stage / role / sub
            rec = js(path / "receipt.json")
            assert rec["source_lock_sha256"] == source and rec["fit_rows_sha256"] == ahash(fr)
            assert rec["query_rows_sha256"] == (None if qr is None else ahash(qr))
            assert rec["n_fit_rows"] == len(fr) and rec["n_fit_people"] == len(
                np.unique(data["patient"][fr])
            )
            if qr is not None:
                assert not set(data["patient"][fr]) & set(data["patient"][qr])
            for p, h in rec["files"].items():
                assert sha(path / p) == h
            bank = nz(path / "models.npz")
            models = {name: model_from(bank, name) for name in rec["models"]}
            assert len(models) == 481 and len(bank) == sum(len(m) for m in models.values())
            old = {}
            for label, parent in (("a", pa), ("g", pg)):
                rr = js(parent / stage / role / sub / "receipt.json")
                assert rr["fit_rows_sha256"] == ahash(fr)
                for p, h in rr["files"].items():
                    assert sha(parent / stage / role / sub / p) == h
                old[label] = nz(parent / stage / role / sub / "models.npz")
            x, y, p, s, c = (data[k][fr] for k in ("color", "target", "patient", "site", "device"))
            w = balanced(p, s)
            close(rec["original_weight_mass"], w.sum(), 2e-10)
            two = len(np.unique(c)) > 1
            assert rec["gate_fits"] == int(two) and rec["new_coefficient_solutions"] == (
                468 if two else 234
            )
            assert rec["gram_decompositions"] == (78 if two else 39)
            for key in (
                "new_coefficient_solutions",
                "gram_decompositions",
                "basis_preparations",
                "exact_width_calculations",
                "covariance_eigendecompositions",
                "gate_fits",
                "auxiliary_baseline_solves",
                "auxiliary_theta_solves",
            ):
                checks[key] += rec[key]
            for op in rec["operations"]:
                assert len(op["solutions"]) == 6
                for v in op["solutions"]:
                    assert np.isfinite(
                        [v["normal_residual"], v["objective"], v["objective_minus_zero"]]
                    ).all()
                    assert v["normal_residual"] < 1e-8 and v["objective_minus_zero"] <= 1e-8
                    maxima["normal_residual"] = max(maxima["normal_residual"], v["normal_residual"])
            template = models[ident("norm_static", 17, "raw", 0.1)]
            validate_normalizer(template, x, y)
            svd = spectrum(norm(template, x), w)
            ev = np.asarray(rec["covariance_eigenvalues"])
            delta = float(np.abs(svd["eigenvalues"] - ev).max())
            maxima["covariance_spectrum"] = max(maxima["covariance_spectrum"], delta)
            np.testing.assert_allclose(ev, svd["eigenvalues"], atol=1e-10, rtol=1e-8)
            close(rec["covariance_trace"], svd["trace"], 1e-10)
            checks["covariance_svd_checks"] += 1
            kernels = {}
            raw_kernel = None
            basis_info = {(r["representation"], r["seed"]): r for r in rec["bases"]}
            for representation, d, tau in REPS:
                rep_model = models[ident("norm_static", 17, representation, 0.1)]
                if representation != "raw":
                    expected = projector(svd, d, tau)
                    diff = float(np.abs(rep_model["projection"] - expected["projection"]).max())
                    maxima["projection_parameter"] = max(maxima["projection_parameter"], diff)
                    # Eigenvectors inside a repeated-eigenvalue subspace may rotate.
                    # The induced squared-distance operator, not those arbitrary
                    # coordinates, is the invariant projection construction.
                    actual_p, expected_p = (
                        rep_model["projection"].astype(np.float64),
                        expected["projection"].astype(np.float64),
                    )
                    actual_metric, expected_metric = (
                        actual_p @ actual_p.T,
                        expected_p @ expected_p.T,
                    )
                    metric_diff = float(
                        np.linalg.norm(actual_metric - expected_metric)
                        / max(np.linalg.norm(expected_metric), 1)
                    )
                    maxima["projection_metric_relative"] = max(
                        maxima["projection_metric_relative"], metric_diff
                    )
                    assert metric_diff <= 2e-6
                    np.testing.assert_allclose(
                        rep_model["projection_mean"], expected["projection_mean"], atol=1e-6, rtol=0
                    )
                    checks["projection_checks"] += 1
                q = latent(rep_model, x)
                width = float(np.float32(direct_width(q)))
                maxima["width"] = max(maxima["width"], abs(width - float(rep_model["width"])))
                assert width == float(rep_model["width"])
                checks["width_checks"] += 1
                kernel = direct_kernel(q, q, width)
                if representation == "raw":
                    raw_kernel = kernel
                if representation == "d36_t1":
                    drift = float(np.abs(kernel - raw_kernel).max())
                    maxima["full_rotation_fit_kernel"] = max(
                        maxima["full_rotation_fit_kernel"], drift
                    )
                    geometry.append(
                        dict(role=role, fold=fold, rotation_kernel_max_difference=drift)
                    )
                for seed in SEEDS:
                    ids, _, _ = select_landmarks(kernel, w, "rpchol", 128, seed)
                    bi = basis_info[(representation, seed)]
                    np.testing.assert_array_equal(ids, bi["fit_center_indices"])
                    m = models[ident("norm_static", seed, representation, 0.1)]
                    np.testing.assert_array_equal(m["centers"], q[ids].astype(np.float32))
                    assert m["centers"].shape == (bi["actual_centers"], d)
                    checks["dense_landmark_checks"] += 1
                    if qr is not None:
                        kernels[(representation, seed)] = direct_kernel(
                            latent(m, data["color"][qr]), m["centers"], width
                        )
            aliases = 0
            for name, m in models.items():
                meta = rec["models"][name]
                Predictor(m)
                assert meta["numeric_bytes"] == sum(v.nbytes for v in m.values())
                if name == "constant":
                    exact(m, dict(constant_lab=np.average(y, axis=0, weights=w).astype(np.float32)))
                    checks["constant_fits"] += 1
                elif meta.get("reference"):
                    old_model = model_from(old[meta["family"][0]], meta["prior_model_id"])
                    exact(m, old_model)
                    raw = models[ident("norm_static", meta["seed"], "raw", 0.1)]
                    for key in ("x_mean", "x_std", "y_mean", "y_std", "width", "centers"):
                        np.testing.assert_array_equal(m[key], raw[key])
                    checks["exact_imported_controls"] += 1
                else:
                    assert name == ident(
                        meta["family"], meta["seed"], meta["representation"], meta["alpha"]
                    )
                    for key in ("x_mean", "x_std", "y_mean", "y_std"):
                        np.testing.assert_array_equal(m[key], template[key])
                    same = models[ident("norm_static", meta["seed"], meta["representation"], 0.1)]
                    for key in (
                        "width",
                        "centers",
                        *(("projection", "projection_mean") if "projection" in m else ()),
                    ):
                        np.testing.assert_array_equal(m[key], same[key])
                    active = two and meta["family"].endswith("joint_soft")
                    assert ("correction" in m) == active
                    if active:
                        assert float(m["rho"]) == 1 and int(m["gate_mode"]) == 1
                    if meta["fallback"]:
                        assert not two
                        exact(
                            m,
                            models[
                                ident(
                                    meta["family"].replace("joint_soft", "static"),
                                    meta["seed"],
                                    meta["representation"],
                                    meta["alpha"],
                                )
                            ],
                        )
                        aliases += 1
                    if meta["representation"] == "raw":
                        old_id = (
                            f"{meta['family']}_s{meta['seed']}_a{ALPHAS.index(meta['alpha'])}_e0"
                        )
                        exact(m, model_from(old["a"], old_id))
                        checks["exact_A_raw_payloads"] += 1
            assert aliases == rec["exact_joint_static_aliases"] == (0 if two else 234)
            checks["exact_aliases"] += aliases
            checks["banks"] += 1
            checks["stored_readouts"] += len(models)
            if qr is not None:
                oof = nz(path / "oof.npz")
                np.testing.assert_array_equal(oof["row_indices"], qr)
                index = np.searchsorted(rows, qr)
                seen.extend(index.tolist())
                for name, m in models.items():
                    meta = rec["models"][name]
                    k = (
                        None
                        if name == "constant"
                        else kernels[(meta.get("representation", "raw"), meta["seed"])]
                    )
                    pred = cached_output(m, k, data["color"][qr])
                    diff = float(np.abs(pred - oof["pred__" + name]).max())
                    maxima["oof_prediction"] = max(maxima["oof_prediction"], diff)
                    assert diff <= 2e-8
                    if name not in merged:
                        merged[name] = np.empty((len(rows), 3))
                    merged[name][index] = pred
                    sizes[name] = max(sizes.get(name, 0), meta["numeric_bytes"])
                    checks["oof_query_rows"] += len(qr)
                if role == "mixed" and fold == 0:
                    for family in FAMILIES:
                        for seed in SEEDS:
                            name = ident(family, seed, "d8_t01", 0.1)
                            reference, info = refit(
                                models[name], x, y, p, s, c, family, 0.1, "d8_t01", seed
                            )
                            pred = predict(reference, data["color"][qr])
                            diff = float(np.abs(pred - oof["pred__" + name]).max())
                            assert diff <= 0.001
                            maxima["qr_prediction"] = max(maxima["qr_prediction"], diff)
                            probes.append(
                                dict(family=family, seed=seed, max_lab_drift=diff, **info)
                            )
                            checks["qr_positive_probes"] += 1
                            checks["qr_query_rows"] += len(qr)
            print(
                f"AUDIT {stage}/{role}/{sub}:481 payloads,13 widths and39 dense pivot paths pass",
                flush=True,
            )
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))
        y, p = data["target"][rows], data["patient"][rows]
        for family in FAMILIES:
            candidates = []
            for order, (rep, _, _) in enumerate(REPS):
                for alpha in ALPHAS:
                    names = [ident(family, s, rep, alpha) for s in SEEDS]
                    ss = [scoring(merged[n], y, p) for n in names]
                    candidates.append(
                        dict(
                            family=family,
                            representation=rep,
                            order=order,
                            alpha=alpha,
                            clean=float(np.mean([r["clean"] for r in ss])),
                            p90=float(np.mean([r["p90"] for r in ss])),
                            numeric_bytes=max(sizes[n] for n in names),
                            seed_scores=ss,
                        )
                    )
            saved = selected["roles"][role][family]
            for a, b in zip(saved["candidates"], candidates, strict=True):
                numeric_equal(a, b)
            raw = sorted(
                (r for r in candidates if r["representation"] == "raw"),
                key=lambda r: (r["clean"], r["p90"], -r["alpha"]),
            )[0]
            numeric_equal(saved["raw_anchor"], raw)
            eligible = [
                r
                for r in candidates
                if r["clean"] <= raw["clean"] + 0.05 and r["p90"] <= raw["p90"] + 0.10
            ]
            choices = dict(
                quality=sorted(
                    candidates,
                    key=lambda r: (r["clean"], r["numeric_bytes"], -r["alpha"], r["order"]),
                )[0],
                compact=sorted(
                    eligible,
                    key=lambda r: (
                        r["numeric_bytes"],
                        r["clean"],
                        r["p90"],
                        -r["alpha"],
                        r["order"],
                    ),
                )[0],
            )
            for policy, choice in choices.items():
                assert (choice["representation"], choice["alpha"]) == (
                    saved["policies"][policy]["representation"],
                    saved["policies"][policy]["alpha"],
                )
                numeric_equal(saved["policies"][policy], choice)
            checks["candidate_score_pairs"] += len(candidates)
            checks["choices"] += 2
        for family in (*CONTROLS, "constant"):
            if family == "constant":
                expected = scoring(merged["constant"], y, p)
            else:
                ss = [scoring(merged[f"{family}_s{s}"], y, p) for s in SEEDS]
                expected = dict(
                    clean=float(np.mean([r["clean"] for r in ss])),
                    p90=float(np.mean([r["p90"] for r in ss])),
                    seed_scores=ss,
                )
            close(selected["references"][role][family], expected)
            checks["reference_score_groups"] += 1
        qr = np.flatnonzero(held)
        x, y, p, s, c = (data[k][qr] for k in ("color", "target", "patient", "site", "device"))
        assert not set(p) & set(data["patient"][rows])
        xx = [transformed(x, t, a) for t, a in SETTINGS]
        bank = nz(run / "final" / role / "bank/models.npz")
        for rec in (r for r in result["records"] if r["role"] == role):
            mp, pp = (
                run / "selected" / role / f"{rec['name']}.npz",
                run / "evaluated" / role / f"{rec['name']}.npz",
            )
            assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
            m, saved = nz(mp), nz(pp)
            exact(m, model_from(bank, rec["model_id"]))
            np.testing.assert_array_equal(saved["row_indices"], qr)
            assert rec["numeric_bytes"] == sum(v.nbytes for v in m.values())
            assert rec["active_gate"] == ("gate_beta" in m)
            call = Predictor(m)
            output = np.array([[call(v) for v in t] for t in xx])
            diff = float(np.abs(output - saved["prediction"]).max())
            maxima["standalone_prediction"] = max(maxima["standalone_prediction"], diff)
            assert diff <= 2e-8
            gates = None if "gate_beta" not in m else np.stack([gate_score(m, t) for t in xx])
            if gates is not None:
                np.testing.assert_allclose(gates, saved["scores"], rtol=0, atol=2e-10)
            met, person_errors = error_summary(output[0], y, p, s)
            close(rec["metrics"], met)
            summaries(rec, output, y, p, c, gates)
            pe[(role, rec["family"], rec["policy"], rec["seed"])] = person_errors
            if rec["family"] in FAMILIES:
                chosen = selected["roles"][role][rec["family"]]["policies"][rec["policy"]]
                assert (rec["representation"], rec["alpha"]) == (
                    chosen["representation"],
                    chosen["alpha"],
                )
                reference, info = refit(
                    m,
                    *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
                    rec["family"],
                    rec["alpha"],
                    rec["representation"],
                    rec["seed"],
                )
                ref_prediction = np.stack([predict(reference, t) for t in xx])
                drift = float(np.abs(ref_prediction - saved["prediction"]).max())
                assert drift <= 0.001
                maxima["qr_prediction"] = max(maxima["qr_prediction"], drift)
                refs.append(dict(role=role, name=rec["name"], max_lab_drift=drift, **info))
                checks["qr_selected_refits"] += 1
                checks["qr_query_rows"] += 33 * len(qr)
                # Reconstruct matched exact A eta0 reference only from its already saved output.
                prior_name = f"{rec['family']}_clean_s{rec['seed']}.npz"
                a_path = pa / "evaluated" / role / prior_name
                a_record = next(
                    v
                    for v in js(pa / "results.json")["records"]
                    if v["role"] == role and v["name"] + ".npz" == prior_name
                )
                assert sha(a_path) == a_record["prediction_sha256"]
                a_saved = nz(a_path)
                np.testing.assert_array_equal(a_saved["row_indices"], qr)
                pe[(role, "raw_" + rec["family"], "reference", rec["seed"])] = error_summary(
                    a_saved["prediction"][0], y, p, s
                )[1]
            elif rec["family"] != "constant":
                if rec["family"].startswith("g_"):
                    parent, name = pg, f"{rec['family'][2:]}_s{rec['seed']}.npz"
                else:
                    parent = pa
                    name = f"{rec['family'][2:].replace('joint_guarded', 'joint_soft')}_guarded_s{rec['seed']}.npz"
                exact(m, nz(parent / "selected" / role / name))
                prior = nz(parent / "evaluated" / role / name)
                np.testing.assert_array_equal(prior["row_indices"], qr)
                expected = prior["prediction"] if parent == pa else prior["prediction"][None]
                np.testing.assert_allclose(
                    output if parent == pa else output[:1], expected, rtol=0, atol=2e-8
                )
            checks["selected_models"] += 1
            checks["final_query_rows"] += 33 * len(qr)
            checks["final_transform_cases"] += 33
            checks["dose_cases"] += 4
        print(
            f"AUDIT {role}:24 independent refits and37x33 actual standalone cases pass", flush=True
        )
    expected_counts = dict(
        banks=12,
        stored_readouts=5772,
        exact_A_raw_payloads=432,
        exact_aliases=1872,
        exact_imported_controls=144,
        constant_fits=12,
        covariance_svd_checks=12,
        projection_checks=144,
        width_checks=156,
        dense_landmark_checks=468,
        oof_query_rows=817700,
        candidate_score_pairs=468,
        reference_score_groups=15,
        choices=24,
        selected_models=111,
        final_query_rows=1462758,
        final_transform_cases=3663,
        dose_cases=444,
        qr_selected_refits=72,
        qr_positive_probes=12,
        new_coefficient_solutions=3744,
        gram_decompositions=624,
        basis_preparations=468,
        exact_width_calculations=156,
        covariance_eigendecompositions=12,
        gate_fits=4,
        auxiliary_baseline_solves=36,
        auxiliary_theta_solves=36,
    )
    for k, v in expected_counts.items():
        assert checks[k] == v, (k, checks[k], v)
    rng = np.random.default_rng(84291)
    paired = []
    for role in masks:
        for family in FAMILIES:
            for policy in ("quality", "compact"):
                delta = np.mean(
                    [
                        pe[(role, family, policy, s)] - pe[(role, "raw_" + family, "reference", s)]
                        for s in SEEDS
                    ],
                    0,
                )
                sample = rng.integers(0, len(delta), (20000, len(delta)))
                paired.append(
                    dict(
                        role=role,
                        family=family,
                        policy=policy,
                        reference="A clean same family",
                        people=len(delta),
                        improved_people=int((delta < 0).sum()),
                        mean_difference=float(delta.mean()),
                        fixed_prediction_person_bootstrap_95=np.quantile(
                            delta[sample].mean(1), [0.025, 0.975]
                        ).tolist(),
                    )
                )
    deps = (
        "scripts/chromaseed_projection_reference.py",
        "tests/test_chromaseed_projection_reference.py",
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
    )
    write_json(
        out / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=source,
            selection_sha256=sha(run / "selections.json"),
            results_sha256=sha(run / "results.json"),
            audit_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in deps},
            checks=checks,
            maxima=maxima,
            independent_selected_refits=refs,
            positive_probes=probes,
            rotation_geometry=geometry,
            paired=paired,
            elapsed_seconds=time.perf_counter() - start,
            scope="Independent weighted-design SVD covariance/projection, exhaustive widths and dense pivot paths, all saved-payload OOF and standalone final outputs/metrics/policies.72 selected and12 positive refits recompute projection/width/landmarks/gate/color metric via SVD/dense/QR; not every grid coefficient refitted. Frozen split/CIEDE2000 reused. Bootstrap is descriptive for fixed historical predictions.",
        ),
    )
    print("X AUDIT PASS", flush=True)


if __name__ == "__main__":
    main()
