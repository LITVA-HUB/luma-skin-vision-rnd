"""Independent direct predictions, person selection and QR/SVD A refits."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_reference import refit
from chromaseed_gate_stability_audit import close, metric, score, transformed
from chromaseed_gated_audit import direct_predict, model_from
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

from luma_skin_vision.color import delta_e00

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
SEEDS, ALPHAS, ETAS = (17, 29, 43), (0.1, 1.0, 10.0), (0.0, 0.25, 0.5, 0.75)
CONTROLS = ("g_norm_base", "g_norm_soft", "g_perceptual_base", "g_perceptual_soft")
ANCHORS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
DOSES = (1 / 255, 4 / 255, 16 / 255, 64 / 255)
SETTINGS = [(0.0, np.zeros(3), -1)] + [(t, a, j) for t in DOSES for j, a in enumerate(ANCHORS)]


def ident(family, seed, alpha, eta):
    return f"{family}_s{seed}_a{ALPHAS.index(alpha)}_e{ETAS.index(eta)}"


def exact(a, b):
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype
        np.testing.assert_array_equal(a[key], b[key])


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def cached_prediction(model, xx, kernels):
    if "constant_lab" in model:
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(xx), 3)).copy()
    value = kernels @ model["coefficient"].astype(np.float64)
    if "correction" in model:
        assert int(model["gate_mode"]) == 1
        gate = np.clip(score(model, xx), -1, 1)
        value += (
            float(model["rho"]) * gate[:, None] * (kernels @ model["correction"].astype(np.float64))
        )
    return value * model["y_std"] + model["y_mean"]


def scores(pred, target, person):
    error = delta_e00(pred, target[None])
    clean, worst = error[0], np.max(error[1:], axis=0)
    people = np.unique(person)
    return dict(
        clean=float(sum(clean[person == p].mean() for p in people) / len(people)),
        robust=float(sum(worst[person == p].mean() for p in people) / len(people)),
    )


def validate_normalizer(model, x, y):
    for prefix, values in (("x", x), ("y", y)):
        values = values.astype(np.float64)
        np.testing.assert_array_equal(model[prefix + "_mean"], values.mean(0).astype(np.float32))
        np.testing.assert_array_equal(
            model[prefix + "_std"], np.maximum(values.std(0), 1e-6).astype(np.float32)
        )


def summaries(rec, output, target, person, camera, gates):
    error = delta_e00(output, target[None])
    drift = delta_e00(output, output[0][None])
    flips = None if gates is None else (gates >= 0) != (gates[0:1] >= 0)
    for i, ((t, a, j), saved) in enumerate(zip(SETTINGS, rec["transforms"], strict=True)):
        close(saved["dose"], t)
        assert saved["anchor_index"] == j
        np.testing.assert_array_equal(saved["anchor"], a)
        for key, v in (
            ("error", error[i]),
            ("drift", drift[i]),
            ("error_change", error[i] - error[0]),
        ):
            close(saved[key], metric(v, person, camera))
        close(saved["sign_flip"], None if flips is None else metric(flips[i], person, camera))
    for i, t in enumerate(DOSES):
        saved, ix = rec["doses"][i], np.arange(1 + 8 * i, 9 + 8 * i)
        close(saved["dose"], t)
        worst = error[ix].max(0)
        for key, v in (
            ("worst_error", worst),
            ("worst_drift", drift[ix].max(0)),
            ("worst_error_change", worst - error[0]),
        ):
            close(saved[key], metric(v, person, camera))
        close(
            saved["any_sign_flip"],
            None if flips is None else metric(flips[ix].any(0), person, camera),
        )


def main():
    parser = argparse.ArgumentParser()
    for name in ("run", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run, args.output
    parent = ROOT / "experiments/runs/chromaseed_gated_v1"
    start = time.perf_counter()
    lock, selected, result = (
        js(run / name) for name in ("source_lock.json", "selections.json", "results.json")
    )
    source = sha(run / "source_lock.json")
    assert source == selected["source_lock_sha256"] == result["source_lock_sha256"]
    assert sha(run / "selections.json") == result["selection_sha256"]
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH == lock["cache_sha256"]
    for path, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / path) == h, path
    assert len(lock["sources"]) == 54 and len(lock["input_sha256"]) == 101
    with np.load(args.cache, allow_pickle=False) as arr:
        data = {k: arr[k] for k in ("color", "target", "patient", "site", "device")}
    masks = roles(data["patient"], data["device"])
    checks = dict(
        banks=0,
        stored_readouts=0,
        exact_aliases=0,
        exact_G_payloads=0,
        constant_fits=0,
        oof_query_rows=0,
        candidate_score_pairs=0,
        reference_score_groups=0,
        choices=0,
        selected_models=0,
        final_transform_cases=0,
        final_query_rows=0,
        dose_cases=0,
        qr_selected_refits=0,
        qr_positive_probes=0,
        qr_query_rows=0,
        new_coefficient_solutions=0,
        gram_decompositions=0,
        basis_preparations=0,
        auxiliary_baseline_solves=0,
        auxiliary_theta_solves=0,
        gate_fits=0,
    )
    maxima = dict(
        oof_prediction=0.0,
        standalone_prediction=0.0,
        qr_prediction=0.0,
        source_weight_mass=0.0,
        normal_residual=0.0,
        objective_minus_zero=-np.inf,
    )
    zero_payload_matches = 0
    gram_clips = 0
    probes, references, person_errors = [], [], {}
    for role, (fit, held) in masks.items():
        rows = np.flatnonzero(fit)
        folds = folds_for(data["patient"][rows], data["device"][rows])
        merged, seen = {}, []
        for fold in (*range(3), None):
            fr = rows if fold is None else rows[folds != fold]
            qr = None if fold is None else rows[folds == fold]
            stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
            directory = run / stage / role / sub
            receipt = js(directory / "receipt.json")
            assert receipt["source_lock_sha256"] == source
            assert receipt["fit_rows_sha256"] == row_hash(fr)
            assert receipt["query_rows_sha256"] == (None if qr is None else row_hash(qr))
            assert receipt["n_fit_rows"] == len(fr)
            assert receipt["n_fit_people"] == len(np.unique(data["patient"][fr]))
            if qr is not None:
                assert not set(data["patient"][fr]) & set(data["patient"][qr])
            for filename, h in receipt["files"].items():
                assert sha(directory / filename) == h
            bank = nz(directory / "models.npz")
            old_dir = parent / stage / role / sub
            old_receipt = js(old_dir / "receipt.json")
            assert old_receipt["fit_rows_sha256"] == row_hash(fr)
            assert sha(old_dir / "models.npz") == old_receipt["files"]["models.npz"]
            old = nz(old_dir / "models.npz")
            models = {name: model_from(bank, name) for name in receipt["models"]}
            assert len(models) == 157
            assert len(bank) == sum(len(m) for m in models.values())
            x, y, person, site, camera = (
                data[k][fr] for k in ("color", "target", "patient", "site", "device")
            )
            w = balanced(person, site)
            close(receipt["original_weight_mass"], w.sum(), 2e-10)
            two = len(np.unique(camera)) > 1
            assert receipt["gate_fits"] == int(two)
            assert receipt["gram_decompositions"] == (24 if two else 12)
            assert receipt["new_coefficient_solutions"] == (144 if two else 72)
            for key in (
                "new_coefficient_solutions",
                "gram_decompositions",
                "basis_preparations",
                "auxiliary_baseline_solves",
                "auxiliary_theta_solves",
                "gate_fits",
            ):
                checks[key] += receipt[key]
            assert len(receipt["operations"]) == receipt["gram_decompositions"]
            for op in receipt["operations"]:
                gram_clips += op["gram_eigenvalues_clipped"]
                assert len(op["solutions"]) == 6
                for r in op["solutions"]:
                    assert np.isfinite(
                        [r["objective"], r["objective_minus_zero"], r["normal_residual"]]
                    ).all()
                    assert r["objective_minus_zero"] <= 1e-8 and r["normal_residual"] < 1e-8
                    maxima["normal_residual"] = max(maxima["normal_residual"], r["normal_residual"])
                    maxima["objective_minus_zero"] = max(
                        maxima["objective_minus_zero"], r["objective_minus_zero"]
                    )
            templates = {s: model_from(old, f"norm_base_s{s}") for s in SEEDS}
            for template in templates.values():
                validate_normalizer(template, x, y)
            aliases = 0
            for name, m in models.items():
                meta = receipt["models"][name]
                if name == "constant":
                    exact(m, dict(constant_lab=np.average(y, axis=0, weights=w).astype(np.float32)))
                    checks["constant_fits"] += 1
                else:
                    Predictor(m)  # Reject malformed/dtype/finite payloads for every configuration.
                    template = templates[meta["seed"]]
                    for key in ("x_mean", "x_std", "y_mean", "y_std", "width", "centers"):
                        np.testing.assert_array_equal(m[key], template[key])
                    if meta.get("reference"):
                        exact(m, model_from(old, meta["prior_model_id"]))
                        checks["exact_G_payloads"] += 1
                    else:
                        assert name == ident(
                            meta["family"], meta["seed"], meta["alpha"], meta["eta"]
                        )
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
                                        meta["alpha"],
                                        meta["eta"],
                                    )
                                ],
                            )
                            aliases += 1
                        if (
                            meta["family"].endswith("static")
                            and meta["alpha"] == 0.1
                            and meta["eta"] == 0
                        ):
                            baseline = model_from(
                                old, f"{meta['family'].split('_')[0]}_base_s{meta['seed']}"
                            )
                            zero_payload_matches += int(
                                set(m) == set(baseline)
                                and all(np.array_equal(m[k], baseline[k]) for k in m)
                            )
            assert aliases == receipt["exact_joint_static_aliases"] == (0 if two else 72)
            checks["exact_aliases"] += aliases
            checks["banks"] += 1
            checks["stored_readouts"] += len(models)
            if qr is not None:
                saved = nz(directory / "oof.npz")
                np.testing.assert_array_equal(saved["row_indices"], qr)
                xx = [
                    transformed(data["color"][qr], t, a)
                    for t, a, _ in (SETTINGS[:1] + SETTINGS[9:17])
                ]
                kernels = {
                    s: [direct_kernel(norm(m, v), m["centers"], float(m["width"])) for v in xx]
                    for s, m in templates.items()
                }
                index = np.searchsorted(rows, qr)
                seen.extend(index.tolist())
                for name, m in models.items():
                    seed = receipt["models"][name]["seed"]
                    kk = [None] * 9 if seed is None else kernels[seed]
                    pred = np.stack(
                        [cached_prediction(m, v, k) for v, k in zip(xx, kk, strict=True)]
                    )
                    diff = float(np.abs(pred - saved["pred__" + name]).max())
                    maxima["oof_prediction"] = max(maxima["oof_prediction"], diff)
                    assert diff < 2e-8
                    if name not in merged:
                        merged[name] = np.empty((9, len(rows), 3))
                    merged[name][:, index] = pred
                    checks["oof_query_rows"] += 9 * len(qr)
                if role == "mixed" and fold == 0:
                    for family in FAMILIES:
                        for seed in SEEDS:
                            name = ident(family, seed, 0.1, 0.75)
                            reference, info = refit(
                                models[name], x, y, person, site, camera, family, 0.1, 0.75
                            )
                            pred = np.stack([direct_predict(reference, v) for v in xx])
                            diff = float(np.abs(pred - saved["pred__" + name]).max())
                            assert diff <= 0.001
                            maxima["qr_prediction"] = max(maxima["qr_prediction"], diff)
                            maxima["source_weight_mass"] = max(
                                maxima["source_weight_mass"], info["source_weight_mass_drift"]
                            )
                            checks["qr_positive_probes"] += 1
                            checks["qr_query_rows"] += 9 * len(qr)
                            probes.append(
                                dict(
                                    family=family,
                                    seed=seed,
                                    alpha=0.1,
                                    eta=0.75,
                                    max_lab_drift=diff,
                                    **info,
                                )
                            )
            print(
                f"AUDIT {stage}/{role}/{sub}:157 payloads, controls, fit boundaries and direct outputs pass",
                flush=True,
            )
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))
        y, person = data["target"][rows], data["patient"][rows]
        for family in FAMILIES:
            saved = selected["roles"][role][family]
            candidates = []
            for eta in ETAS:
                for alpha in ALPHAS:
                    ss = [
                        scores(merged[ident(family, seed, alpha, eta)], y, person) for seed in SEEDS
                    ]
                    candidates.append(
                        dict(
                            family=family,
                            alpha=alpha,
                            eta=eta,
                            clean=float(np.mean([r["clean"] for r in ss])),
                            robust=float(np.mean([r["robust"] for r in ss])),
                            seed_scores=ss,
                        )
                    )
            # Compare numbers separately: the old numeric helper deliberately has no string case.
            for expected, actual in zip(saved["candidates"], candidates, strict=True):
                assert expected["family"] == actual["family"]
                close(
                    {k: v for k, v in expected.items() if k != "family"},
                    {k: v for k, v in actual.items() if k != "family"},
                )
            anchor = min(r["clean"] for r in candidates if r["eta"] == 0)
            close(saved["zero_eta_anchor_clean"], anchor)
            assert saved["clean_allowance"] == 0.05
            feasible = [r for r in candidates if r["clean"] <= anchor + 0.05]
            choices = dict(
                clean=sorted(candidates, key=lambda r: (r["clean"], r["eta"], -r["alpha"]))[0],
                guarded=sorted(
                    feasible, key=lambda r: (r["robust"], r["clean"], r["eta"], -r["alpha"])
                )[0],
            )
            for policy, chosen in choices.items():
                actual = saved["policies"][policy]
                assert (actual["alpha"], actual["eta"], actual["family"]) == (
                    chosen["alpha"],
                    chosen["eta"],
                    family,
                )
                close(
                    {k: v for k, v in actual.items() if k != "family"},
                    {k: v for k, v in chosen.items() if k != "family"},
                )
            checks["candidate_score_pairs"] += len(candidates)
            checks["choices"] += 2
        for family in (*CONTROLS, "constant"):
            if family == "constant":
                value = scores(merged[family], y, person)
            else:
                ss = [scores(merged[f"{family}_s{s}"], y, person) for s in SEEDS]
                value = dict(
                    clean=float(np.mean([r["clean"] for r in ss])),
                    robust=float(np.mean([r["robust"] for r in ss])),
                    seed_scores=ss,
                )
            close(selected["references"][role][family], value)
            checks["reference_score_groups"] += 1
        qr = np.flatnonzero(held)
        x, y, person, site, camera = (
            data[k][qr] for k in ("color", "target", "patient", "site", "device")
        )
        assert not set(person) & set(data["patient"][rows])
        xx = [transformed(x, t, a) for t, a, _ in SETTINGS]
        final_bank = nz(run / "final" / role / "bank/models.npz")
        for rec in (r for r in result["records"] if r["role"] == role):
            name = rec["name"]
            mp, pp = (
                run / "selected" / role / f"{name}.npz",
                run / "evaluated" / role / f"{name}.npz",
            )
            assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
            m, saved = nz(mp), nz(pp)
            exact(m, model_from(final_bank, rec["model_id"]))
            np.testing.assert_array_equal(saved["row_indices"], qr)
            assert sum(v.nbytes for v in m.values()) == rec["numeric_bytes"]
            assert ("gate_beta" in m) == rec["active_gate"]
            if "constant_lab" in m:
                output = np.broadcast_to(
                    m["constant_lab"].astype(np.float64), (33, len(qr), 3)
                ).copy()
            else:
                call = Predictor(m)
                output = np.array([[call(v) for v in t] for t in xx])
            diff = float(np.abs(output - saved["prediction"]).max())
            maxima["standalone_prediction"] = max(maxima["standalone_prediction"], diff)
            assert diff < 2e-8
            gates = np.stack([score(m, t) for t in xx]) if "gate_beta" in m else None
            if gates is not None:
                np.testing.assert_allclose(gates, saved["scores"], rtol=0, atol=2e-10)
            met, pe = error_summary(output[0], y, person, site)
            close(rec["metrics"], met)
            summaries(rec, output, y, person, camera, gates)
            person_errors[(role, rec["family"], rec["policy"], rec["seed"])] = pe
            if rec["family"] in FAMILIES:
                choice = selected["roles"][role][rec["family"]]["policies"][rec["policy"]]
                assert (rec["alpha"], rec["eta"]) == (choice["alpha"], choice["eta"])
                assert rec["model_id"] == ident(
                    rec["family"], rec["seed"], rec["alpha"], rec["eta"]
                )
                reference, info = refit(
                    m,
                    *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
                    rec["family"],
                    rec["alpha"],
                    rec["eta"],
                )
                pred = np.stack([direct_predict(reference, t) for t in xx])
                drift = float(np.abs(pred - saved["prediction"]).max())
                assert drift <= 0.001
                maxima["qr_prediction"] = max(maxima["qr_prediction"], drift)
                maxima["source_weight_mass"] = max(
                    maxima["source_weight_mass"], info["source_weight_mass_drift"]
                )
                references.append(dict(role=role, name=name, max_lab_drift=drift, **info))
                checks["qr_selected_refits"] += 1
                checks["qr_query_rows"] += 33 * len(qr)
            elif rec["family"] in CONTROLS:
                old_name = f"{rec['family'][2:]}_s{rec['seed']}.npz"
                exact(m, nz(parent / "selected" / role / old_name))
                prior = nz(parent / "evaluated" / role / old_name)
                np.testing.assert_array_equal(qr, prior["row_indices"])
                np.testing.assert_allclose(output[0], prior["prediction"], atol=2e-8, rtol=0)
            checks["selected_models"] += 1
            checks["final_transform_cases"] += 33
            checks["final_query_rows"] += 33 * len(qr)
            checks["dose_cases"] += 4
        print(
            f"AUDIT {role}:24 QR refits,37x33 standalone outputs and both inner policies pass",
            flush=True,
        )
    expected = dict(
        banks=12,
        stored_readouts=1884,
        exact_aliases=576,
        exact_G_payloads=144,
        constant_fits=12,
        oof_query_rows=2402100,
        candidate_score_pairs=144,
        reference_score_groups=15,
        choices=24,
        selected_models=111,
        final_transform_cases=3663,
        final_query_rows=1462758,
        dose_cases=444,
        qr_selected_refits=72,
        qr_positive_probes=12,
        new_coefficient_solutions=1152,
        gram_decompositions=192,
        basis_preparations=36,
        auxiliary_baseline_solves=36,
        auxiliary_theta_solves=36,
        gate_fits=4,
    )
    for key, value in expected.items():
        assert checks[key] == value, (key, checks[key], value)
    paired = []
    rng = np.random.default_rng(94103)
    for role in masks:
        pairs = [(f, "guarded", f, "clean") for f in FAMILIES]
        pairs += [
            (f"{b}_joint_soft", "clean", f"g_{b}_soft", "reference") for b in ("norm", "perceptual")
        ]
        pairs += [
            (f"{b}_joint_soft", "clean", f"{b}_static", "clean") for b in ("norm", "perceptual")
        ]
        for a, ap, b, bp in pairs:
            d = np.mean(
                [person_errors[(role, a, ap, s)] - person_errors[(role, b, bp, s)] for s in SEEDS],
                0,
            )
            draws = rng.integers(0, len(d), size=(20000, len(d)))
            ci = np.quantile(d[draws].mean(1), [0.025, 0.975])
            paired.append(
                dict(
                    role=role,
                    candidate=a,
                    candidate_policy=ap,
                    reference=b,
                    reference_policy=bp,
                    mean_difference=float(d.mean()),
                    people=len(d),
                    improved_people=int((d < 0).sum()),
                    fixed_prediction_person_bootstrap_95=ci.tolist(),
                )
            )
    deps = (
        "scripts/chromaseed_affine_reference.py",
        "tests/test_chromaseed_affine_reference.py",
        "scripts/chromaseed_gate_stability_audit.py",
        "scripts/chromaseed_gated_audit.py",
        "scripts/chromaseed_gated_numpy.py",
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
            zero_eta_static_payloads_exact_G=zero_payload_matches,
            recorded_negative_gram_eigenvalues=gram_clips,
            independent_selected_refits=references,
            positive_probes=probes,
            paired=paired,
            elapsed_seconds=time.perf_counter() - start,
            scope="All banks/OOF/direct predictions, frozen policies and all final standalone outputs/metrics. Independent QR/SVD refits cover72 selected new instances plus12 positive probes, not every grid fit. Shared frozen splits/CIEDE2000; normalizers checked on original fit rows, basis pinned to audited G. Bootstrap is descriptive for fixed historically reused predictions, not fresh accuracy confidence.",
        ),
    )
    print("A AUDIT PASS", flush=True)


if __name__ == "__main__":
    main()
