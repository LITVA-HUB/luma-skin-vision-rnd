"""Independent direct-kernel/analytic-metric/SVD audit of G readouts and gates."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from chromaseed_fast_kernel import get_model
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced, reference_readouts
from chromaseed_perceptual_reference import analytic_tensor, augmented_svd
from chromaseed_refine_audit import error_summary
from scipy.linalg import lstsq
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
BASES = ("norm", "perceptual")
FAMILIES = tuple(f"{b}_{r}" for b in BASES for r in ("base", "uniform", "soft", "hard"))
SEEDS = (17, 29, 43)


def ident(family, seed, alpha=10.0, rho=0.0):
    if rho == 0.0 or family.endswith("_base"):
        return f"{family.split('_')[0]}_base_s{seed}"
    return f"{family}_s{seed}_l{(0.1, 1.0, 10.0).index(alpha)}_r{(0.25, 0.5, 1.0).index(rho)}"


def model_from(arrays, name):
    return {k.split("__", 1)[1]: v for k, v in arrays.items() if k.split("__", 1)[0] == name}


def direct_predict(model, x):
    z = norm(model, x)
    k = direct_kernel(z, model["centers"], float(model["width"]))
    y = k @ model["coefficient"].astype(np.float64)
    if "correction" in model:
        assert int(model["gate_mode"]) in (1, 2)
        score = np.sum(z * model["gate_beta"][None, 1:].astype(np.float64), axis=1) + float(
            model["gate_beta"][0]
        )
        signal = (
            np.minimum(np.maximum(score, -1.0), 1.0)
            if int(model["gate_mode"]) == 1
            else np.where(score >= 0, 1.0, -1.0)
        )
        y = y + float(model["rho"]) * signal[:, None] * (k @ model["correction"].astype(np.float64))
    return y * model["y_std"] + model["y_mean"]


def reference_gate(z, person, site, camera):
    w = np.empty(len(person))
    for c in np.unique(camera):
        mask = camera == c
        w[mask] = balanced(person[mask], site[mask]) / (np.sum(mask) * len(np.unique(camera)))
    w /= w.sum()
    mean = np.array([sum(w * col) for col in z.T])
    std = np.maximum(
        np.array([np.sqrt(sum(w * (col - m) ** 2)) for col, m in zip(z.T, mean, strict=True)]), 1e-6
    )
    a = np.column_stack([np.ones(len(z)), (z - mean) / std])
    penalty = np.column_stack([np.zeros(z.shape[1]), np.sqrt(0.1) * np.eye(z.shape[1])])
    theta = lstsq(
        np.vstack([np.sqrt(w)[:, None] * a, penalty]),
        np.r_[np.sqrt(w) * np.where(camera == "SLR", 1.0, -1.0), np.zeros(z.shape[1])],
        lapack_driver="gelsd",
    )[0]
    beta = theta[1:] / std
    return np.r_[theta[0] - mean @ beta, beta].astype(np.float32)


def refit(base, x, y, person, site, camera, family, alpha, rho):
    which, route = family.split("_")
    weights = balanced(person, site)
    old = "norm_mse" if which == "norm" else "constant_de2"
    step = 0 if which == "norm" else 1
    bcoef, _ = reference_readouts(base, x, y, weights, old, 0.1, step)
    result = {**base, "coefficient": bcoef[step]}
    if rho == 0.0 or route == "base" or (route in ("soft", "hard") and len(np.unique(camera)) < 2):
        return result
    z = norm(base, x)
    columns = direct_kernel(z, base["centers"], float(base["width"]))
    km = direct_kernel(base["centers"], base["centers"], float(base["width"]))
    u, eigen, _ = np.linalg.svd(km, full_matrices=False)
    keep = eigen > eigen.max() * 1e-8
    white = u[:, keep] / np.sqrt(eigen[keep])
    design = columns @ white
    beta = None
    if route != "uniform":
        beta = reference_gate(z, person, site, camera)
        score = z @ beta[1:].astype(np.float64) + float(beta[0])
        signal = np.clip(score, -1.0, 1.0) if route == "soft" else np.where(score >= 0.0, 1.0, -1.0)
        design = design * signal[:, None]
    target = (y.astype(np.float64) - base["y_mean"]) / base["y_std"]
    residual = target - columns @ result["coefficient"].astype(np.float64)
    metric = np.eye(3)
    if which == "perceptual":
        raw = analytic_tensor(y)
        std = base["y_std"].astype(np.float64)
        g = raw * std[None, :, None] * std[None, None, :]
        g /= np.average(np.trace(g, axis1=1, axis2=2), weights=weights) / 3
        metric = np.average(g, axis=0, weights=weights)
    correction = (
        white
        @ augmented_svd(design, residual, np.broadcast_to(metric, (len(x), 3, 3)), weights, alpha)
    ).astype(np.float32)
    if route == "uniform":
        result["coefficient"] = (
            result["coefficient"].astype(np.float64) + rho * correction.astype(np.float64)
        ).astype(np.float32)
    else:
        result.update(
            correction=correction,
            gate_beta=beta,
            gate_mode=np.asarray(1 if route == "soft" else 2, np.uint8),
            rho=np.asarray(rho, np.float32),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ("run", "cache", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument(
        "--parent", type=Path, default=ROOT / "experiments/runs/chromaseed_perceptual_v1"
    )
    args = parser.parse_args()
    run = args.run
    started = time.perf_counter()
    lock = js(run / "source_lock.json")
    source = sha(run / "source_lock.json")
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH == lock["cache_sha256"]
    for rel, h in lock["sources"].items():
        assert sha(ROOT / rel) == h, rel
    for rel, h in lock["input_sha256"].items():
        assert sha(ROOT / rel) == h, rel
    for rel, h in lock["parent_bindings"].items():
        assert sha(args.parent / rel) == h, rel
    selected = js(run / "selections.json")
    results = js(run / "results.json")
    assert selected["source_lock_sha256"] == source and results["selection_sha256"] == sha(
        run / "selections.json"
    )
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    checks = dict(
        banks=0,
        stored_readouts=0,
        exact_P_bases=0,
        exact_fallbacks=0,
        oof_query_rows=0,
        choices=0,
        selected_models=0,
        selected_query_rows=0,
        independent_selected_refits=0,
        positive_route_probes=0,
        residual_solutions=0,
        gate_fits=0,
        perceptual_base_solves=0,
    )
    maxima = dict(
        oof_prediction_drift=0.0,
        selected_prediction_drift=0.0,
        independent_refit_drift=0.0,
        probe_drift=0.0,
        objective_increase=0.0,
        normal_residual=0.0,
    )
    paired = []
    probes = []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        rows = np.flatnonzero(fit)
        folds = folds_for(data["patient"][fit], data["device"][fit])
        oof = []
        for fold in (0, 1, 2, None):
            stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
            directory = run / stage / role / sub
            receipt = js(directory / "receipt.json")
            assert receipt["source_lock_sha256"] == source
            for rel, h in receipt["files"].items():
                assert sha(directory / rel) == h, rel
            bank = nz(directory / "models.npz")
            pbank = nz(args.parent / stage / role / sub / "models.npz")
            a = rows if fold is None else rows[folds != fold]
            b = np.flatnonzero(held) if fold is None else rows[folds == fold]
            assert not set(data["patient"][a]) & set(data["patient"][b])
            for base, old, step in [("norm", "norm_mse", 0), ("perceptual", "constant_de2", 1)]:
                for seed in SEEDS:
                    actual = model_from(bank, ident(base + "_base", seed))
                    expected = get_model(pbank, f"{old}_s{seed}_w1_a0_t{step}")
                    assert set(actual) == set(expected)
                    for field in expected:
                        np.testing.assert_array_equal(actual[field], expected[field])
                    checks["exact_P_bases"] += 1
            for name, config in receipt["models"].items():
                model = model_from(bank, name)
                assert all(np.isfinite(v).all() for v in model.values())
                assert all(
                    v.dtype == (np.uint8 if k == "gate_mode" else np.float32)
                    for k, v in model.items()
                )
                if config["fallback"] == "single_camera":
                    assert len(np.unique(data["device"][a])) == 1
                    base = model_from(
                        bank, ident(config["family"].split("_")[0] + "_base", config["seed"])
                    )
                    assert set(model) == set(base)
                    for key in base:
                        np.testing.assert_array_equal(model[key], base[key])
                    checks["exact_fallbacks"] += 1
                checks["stored_readouts"] += 1
            for operation in receipt["operations"]:
                for solve in operation["solutions"]:
                    maxima["objective_increase"] = max(
                        maxima["objective_increase"], solve["objective_minus_zero"]
                    )
                    maxima["normal_residual"] = max(
                        maxima["normal_residual"], solve["normal_relative_residual"]
                    )
                    assert (
                        solve["objective_minus_zero"] <= 1e-8
                        and solve["normal_relative_residual"] < 1e-8
                    )
            for key in ("residual_solutions", "gate_fits", "perceptual_base_solves"):
                checks[key] += receipt[key]
            checks["banks"] += 1
            if fold is not None:
                saved = nz(directory / "oof.npz")
                np.testing.assert_array_equal(saved["row_indices"], b)
                for name in receipt["models"]:
                    pred = direct_predict(model_from(bank, name), data["color"][b])
                    drift = float(np.abs(pred - saved["pred__" + name]).max())
                    assert drift < 1e-7
                    maxima["oof_prediction_drift"] = max(maxima["oof_prediction_drift"], drift)
                    checks["oof_query_rows"] += len(b)
                oof.append(saved)
            if role == "mixed" and fold == 0:
                for base in BASES:
                    for seed in SEEDS:
                        shared = model_from(bank, ident(base + "_base", seed))
                        for route in ("soft", "hard"):
                            family = base + "_" + route
                            actual = model_from(bank, ident(family, seed, 0.1, 1.0))
                            reference = refit(
                                shared,
                                *(
                                    data[k][a]
                                    for k in ("color", "target", "patient", "site", "device")
                                ),
                                family,
                                0.1,
                                1.0,
                            )
                            drift = float(
                                np.abs(
                                    direct_predict(actual, data["color"][b])
                                    - direct_predict(reference, data["color"][b])
                                ).max()
                            )
                            assert drift <= 0.001
                            maxima["probe_drift"] = max(maxima["probe_drift"], drift)
                            probes.append(
                                dict(family=family, seed=seed, max_native_lab_drift=drift)
                            )
                            checks["positive_route_probes"] += 1
        oofrows = np.concatenate([o["row_indices"] for o in oof])
        order = np.argsort(oofrows)
        oofrows = oofrows[order]
        np.testing.assert_array_equal(oofrows, rows)
        for family in FAMILIES:
            cfg = [dict(residual_lambda=10.0, rho=0.0)]
            if not family.endswith("_base"):
                cfg.extend(
                    dict(residual_lambda=a, rho=r)
                    for a in (0.1, 1.0, 10.0)
                    for r in (0.25, 0.5, 1.0)
                )
            for c in cfg:
                values = []
                for seed in SEEDS:
                    pred = np.concatenate(
                        [
                            o["pred__" + ident(family, seed, c["residual_lambda"], c["rho"])]
                            for o in oof
                        ]
                    )[order]
                    values.append(
                        error_summary(
                            pred, data["target"][rows], data["patient"][rows], data["site"][rows]
                        )[0]["person_mean"]
                    )
                c["person_mean"] = float(np.mean(values))
            best = min(cfg, key=lambda c: (c["person_mean"], c["rho"], -c["residual_lambda"]))
            chosen = selected["roles"][role][family]["selected"]
            assert (
                best["residual_lambda"] == chosen["residual_lambda"]
                and best["rho"] == chosen["rho"]
                and abs(best["person_mean"] - chosen["person_mean"]) < 1e-10
            )
            checks["choices"] += 1
        bank = nz(run / "final" / role / "bank/models.npz")
        final_scores = {}
        b = np.flatnonzero(held)
        for record in (r for r in results["records"] if r["role"] == role):
            family, seed = record["family"], record["seed"]
            filename = f"{family}_s{seed}.npz"
            mp, pp = run / "selected" / role / filename, run / "evaluated" / role / filename
            assert sha(mp) == record["model_sha256"] and sha(pp) == record["prediction_sha256"]
            model, saved = nz(mp), nz(pp)
            np.testing.assert_array_equal(saved["row_indices"], b)
            pred = direct_predict(model, data["color"][b])
            drift = float(np.abs(pred - saved["prediction"]).max())
            assert drift < 1e-7
            maxima["selected_prediction_drift"] = max(maxima["selected_prediction_drift"], drift)
            metric, person_score = error_summary(
                pred, data["target"][b], data["patient"][b], data["site"][b]
            )
            for k, v in metric.items():
                if k in record["metrics"]:
                    assert abs(v - record["metrics"][k]) < 1e-9
            assert sum(v.nbytes for v in model.values()) == record["numeric_bytes"]
            assert record["active_gate"] == ("correction" in model)
            final_scores[(family, seed)] = person_score
            shared = model_from(bank, ident(family.split("_")[0] + "_base", seed))
            reference = refit(
                shared,
                *(data[k][rows] for k in ("color", "target", "patient", "site", "device")),
                family,
                record["residual_lambda"],
                record["rho"],
            )
            drift = float(np.abs(pred - direct_predict(reference, data["color"][b])).max())
            assert drift <= 0.001
            maxima["independent_refit_drift"] = max(maxima["independent_refit_drift"], drift)
            checks["selected_models"] += 1
            checks["selected_query_rows"] += len(b)
            checks["independent_selected_refits"] += 1
        rng = np.random.default_rng(2026091307)
        draws = rng.integers(
            len(np.unique(data["patient"][b])), size=(20000, len(np.unique(data["patient"][b])))
        )
        for family in FAMILIES:
            if family.endswith("_base"):
                continue
            base = family.split("_")[0]
            for reference in (base + "_base", base + "_uniform"):
                if reference == family:
                    continue
                difference = np.mean(
                    [final_scores[(family, s)] - final_scores[(reference, s)] for s in SEEDS],
                    axis=0,
                )
                paired.append(
                    dict(
                        role=role,
                        family=family,
                        reference=reference,
                        person_mean_difference=float(difference.mean()),
                        people=len(difference),
                        people_improved=int(np.sum(difference < 0)),
                        descriptive_fixed_prediction_person_range=np.quantile(
                            difference[draws].mean(1), [0.025, 0.975]
                        ).tolist(),
                    )
                )
        print(
            f"AUDITED {role}: full bank checks, eight choices,24 independent selected refits",
            flush=True,
        )
    expected = dict(
        banks=12,
        stored_readouts=2016,
        exact_P_bases=72,
        exact_fallbacks=864,
        oof_query_rows=285600,
        choices=24,
        selected_models=72,
        selected_query_rows=28752,
        independent_selected_refits=72,
        positive_route_probes=12,
        residual_solutions=360,
        gate_fits=4,
        perceptual_base_solves=36,
    )
    assert checks == expected, (checks, expected)
    deps = (
        "scripts/chromaseed_kernel_audit.py",
        "scripts/chromaseed_perceptual_audit.py",
        "scripts/chromaseed_perceptual_reference.py",
        "scripts/chromaseed_refine_audit.py",
        "scripts/skin_local_search_train.py",
        "src/luma_skin_vision/color.py",
    )
    write_json(
        args.output / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=source,
            selection_sha256=sha(run / "selections.json"),
            results_sha256=sha(run / "results.json"),
            audit_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in deps},
            checks=checks,
            maxima=maxima,
            positive_route_probes=probes,
            paired=paired,
            elapsed_seconds=time.perf_counter() - started,
            limits="Historically reused TRAIN; fixed prediction bootstrap is descriptive, correlated roles/fit/selection uncertainty omitted. Independent metric uses existing verified CIEDE2000 and role helpers.",
        ),
    )


if __name__ == "__main__":
    main()
