"""Independent H geometry, selection, real NumPy queries and QR/SVD reconstructions."""

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
from chromaseed_hybrid_numpy import Predictor
from chromaseed_hybrid_reference import Basis, Geometry, predict
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_projection_audit import numeric_equal, scoring
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
LOSSES, SEEDS, ALPHAS = ("norm", "perceptual"), (17, 29, 43), (0.1, 1.0, 10.0)
KINDS, RHOS = ("raw", "projected", "blend", "uniform", "support"), (0.25, 0.5, 1.0)
SETTINGS = [(0.0, np.zeros(3))] + [(t, a) for t in DOSES for a in ANCHORS]


def ident(loss, seed, kind, alpha=0.1, rho=0.0, power=0):
    name = f"{loss}_{kind}_s{seed}"
    if kind == "raw":
        return name
    name += f"_a{ALPHAS.index(alpha)}"
    return name if kind == "projected" else f"{name}_r{RHOS.index(rho)}_p{power}"


def options(kind):
    raw = dict(kind="raw", alpha=0.1, rho=0.0, power=0)
    if kind == "raw":
        return [raw]
    if kind == "projected":
        return [dict(kind=kind, alpha=a, rho=1.0, power=0) for a in ALPHAS]
    return [raw] + [
        dict(kind=kind, alpha=a, rho=r, power=p)
        for a in ALPHAS
        for r in RHOS
        for p in ((1, 4) if kind == "support" else (0,))
    ]


def ahash(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def cached_prediction(m, x, cache):
    if "constant_lab" in m:
        return np.broadcast_to(m["constant_lab"].astype(np.float64), (len(x), 3)).copy()

    def kernel(coords, centers, width, key):
        if key not in cache:
            cache[key] = direct_kernel(coords, centers, width)
        return cache[key]

    raw = norm(m, x)
    pfields = [m[k].tobytes() for k in ("centers", "width")]
    if "projection" in m:
        p, mean = m["projection"].astype(np.float64), m["projection_mean"].astype(np.float64)
        q = ((raw - mean) @ p).astype(np.float32).astype(np.float64)
        pfields += [m["projection"].tobytes(), m["projection_mean"].tobytes()]
    else:
        q = raw
    kr = kernel(q, m["centers"], float(m["width"]), tuple(pfields))
    s = None if "gate_beta" not in m else np.clip(gate_score(m, x), -1, 1)[:, None]
    base = kr @ m["coefficient"].astype(np.float64)
    if s is not None:
        base += s * (kr @ m["correction"].astype(np.float64))
    if "hybrid_mode" in m:
        p, mean = m["latent_projection"].astype(np.float64), m["latent_mean"].astype(np.float64)
        q = ((raw - mean) @ p).astype(np.float32).astype(np.float64)
        centers = ((m["centers"].astype(np.float64) - mean) @ p).astype(np.float32)
        key = tuple(
            pfields + [m[k].tobytes() for k in ("latent_projection", "latent_mean", "latent_width")]
        )
        kp = kernel(q, centers, float(m["latent_width"]), key)
        branch = kp @ m["latent_coefficient"].astype(np.float64)
        if s is not None:
            branch += s * (kp @ m["latent_correction"].astype(np.float64))
        rho = float(m["mix"])
        if int(m["hybrid_mode"]) == 1:
            base = (1 - rho) * base + rho * branch
        else:
            h = (np.sum(kr, axis=1) / kr.shape[1]) ** int(m["support_power"])
            base = base + rho * h[:, None] * branch
    return base * m["y_std"] + m["y_mean"]


def main():
    parser = argparse.ArgumentParser()
    for field in ("run", "cache", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    run, out, start = args.run, args.output, time.perf_counter()
    lock, selections, result = (
        js(run / p) for p in ("source_lock.json", "selections.json", "results.json")
    )
    source = sha(run / "source_lock.json")
    assert source == selections["source_lock_sha256"] == result["source_lock_sha256"]
    assert result["selection_sha256"] == sha(run / "selections.json")
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH == lock["cache_sha256"]
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    assert len(lock["sources"]) == 64 and len(lock["input_sha256"]) == 73
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    checks = dict(
        banks=0,
        stored_readouts=0,
        exact_A_controls=0,
        imported_X_controls=0,
        endpoint_aliases=0,
        covariance_svd=0,
        shared_bases=0,
        widths=0,
        oof_rows=0,
        candidate_scores=0,
        choices=0,
        references=0,
        final_query_rows=0,
        final_models=0,
        dose_summaries=0,
        selected_reconstructions=0,
        forced_positive_reconstructions=0,
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
        oof=0.0,
        standalone=0.0,
        direct_identity=0.0,
        qr_prediction=0.0,
        normal_residual=0.0,
        projection_metric_relative=0.0,
        width=0.0,
    )
    reconstructions, geometry_records, person_errors = [], [], {}
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        rows, held_rows = np.flatnonzero(fit), np.flatnonzero(held)
        folds = folds_for(data["patient"][rows], data["device"][rows])
        merged, sizes, seen = {}, {}, []
        for fold in (*range(3), None):
            fr = rows if fold is None else rows[folds != fold]
            qr = None if fold is None else rows[folds == fold]
            stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
            path = run / stage / role / sub
            rec, bank = js(path / "receipt.json"), nz(path / "models.npz")
            assert rec["source_lock_sha256"] == source and rec["fit_rows_sha256"] == ahash(fr)
            assert rec["query_rows_sha256"] == (None if qr is None else ahash(qr))
            for p, h in rec["files"].items():
                assert sha(path / p) == h
            if qr is not None:
                assert not set(data["patient"][fr]) & set(data["patient"][qr])
            models = {name: model_from(bank, name) for name in rec["models"]}
            assert len(models) == 247 and len(bank) == sum(len(m) for m in models.values())
            original = {}
            for tag, parent in (("a", "chromaseed_affine_v1"), ("x", "chromaseed_projection_v1")):
                pp = ROOT / "experiments/runs" / parent / stage / role / sub
                assert js(pp / "receipt.json")["fit_rows_sha256"] == ahash(fr)
                original[tag] = nz(pp / "models.npz")
            xx, yy, pp, ss, cc = (
                data[k][fr] for k in ("color", "target", "patient", "site", "device")
            )
            g = Geometry(xx, yy, pp, ss, cc)
            checks["covariance_svd"] += 1
            checks["widths"] += 2
            assert rec["n_fit_rows"] == len(fr) and rec["n_fit_people"] == len(np.unique(pp))
            close(rec["original_weight_mass"], g.w.sum(), 2e-10)
            assert rec["gate_fits"] == int(len(np.unique(cc)) > 1)
            for field in (
                "new_coefficient_solutions",
                "gram_decompositions",
                "basis_preparations",
                "exact_width_calculations",
                "covariance_eigendecompositions",
                "gate_fits",
                "auxiliary_baseline_solves",
                "auxiliary_theta_solves",
            ):
                checks[field] += rec[field]
            assert rec["new_coefficient_solutions"] == 78 and rec["gram_decompositions"] == 24
            for op in rec["operations"]:
                for v in op["solutions"]:
                    assert v["normal_residual"] <= 1e-8 and v["objective_minus_zero"] <= 1e-8
                    maxima["normal_residual"] = max(maxima["normal_residual"], v["normal_residual"])
            for name, m in models.items():
                Predictor(m)
                assert rec["models"][name]["numeric_bytes"] == sum(v.nbytes for v in m.values())
                if name != "constant":
                    validate_normalizer(m, xx, yy)
            np.testing.assert_array_equal(
                models["constant"]["constant_lab"],
                np.average(yy, axis=0, weights=g.w).astype(np.float32),
            )
            for seed in SEEDS:
                b = Basis(g, seed)
                bm = next(v for v in rec["bases"] if v["seed"] == seed)
                np.testing.assert_array_equal(b.ids, bm["fit_center_indices"])
                assert (
                    bm["raw_rank"] == b.rwhite.shape[1]
                    and bm["projected_rank"] == b.pwhite.shape[1]
                )
                checks["shared_bases"] += 1
                for loss in LOSSES:
                    raw = models[ident(loss, seed, "raw")]
                    exact(raw, model_from(original["a"], f"{loss}_joint_soft_s{seed}_a0_e0"))
                    checks["exact_A_controls"] += 1
                    exact(
                        models[f"x_fixed_{loss}_s{seed}"],
                        model_from(original["x"], f"{loss}_joint_soft_d16_t05_s{seed}_a0"),
                    )
                    checks["imported_X_controls"] += 1
                    for ai, alpha in enumerate(ALPHAS):
                        proj = models[ident(loss, seed, "projected", alpha)]
                        p = proj["projection"].astype(np.float64)
                        refp = g.projection["projection"].astype(np.float64)
                        drift = float(
                            np.linalg.norm(p @ p.T - refp @ refp.T)
                            / max(np.linalg.norm(refp @ refp.T), 1e-30)
                        )
                        assert drift < 2e-6
                        maxima["projection_metric_relative"] = max(
                            maxima["projection_metric_relative"], drift
                        )
                        stored_q = (
                            (g.raw - proj["projection_mean"].astype(np.float64)) @ p
                        ).astype(np.float32)
                        np.testing.assert_array_equal(proj["centers"], stored_q[b.ids])
                        maxima["width"] = max(
                            maxima["width"],
                            abs(float(proj["width"]) - g.pw),
                            abs(float(raw["width"]) - g.rw),
                        )
                        exact(models[ident(loss, seed, "blend", alpha, 1.0, 0)], proj)
                        checks["endpoint_aliases"] += 1
                    for kind in ("blend", "uniform", "support"):
                        for setting in options(kind)[1:]:
                            m = models[ident(loss, seed, **setting)]
                            if "hybrid_mode" in m:
                                for k, v in raw.items():
                                    np.testing.assert_array_equal(m[k], v)
                requests = []
                if fold is None:
                    for record in result["records"]:
                        if (
                            record["role"] == role
                            and record["policy"] == "inner"
                            and record["seed"] == seed
                        ):
                            requests.append((record, models[record["model_id"]], False))
                elif role == "mixed" and fold == 0:
                    for loss in LOSSES:
                        for kind in ("blend", "uniform", "support"):
                            request = dict(
                                loss=loss,
                                kind=kind,
                                alpha=0.1,
                                rho=0.5,
                                power=4 if kind == "support" else 0,
                            )
                            requests.append(
                                (
                                    request,
                                    models[
                                        ident(
                                            loss,
                                            seed,
                                            **{
                                                k: request[k]
                                                for k in ("kind", "alpha", "rho", "power")
                                            },
                                        )
                                    ],
                                    True,
                                )
                            )
                for setting, expected, probe in requests:
                    fitted = b.make(
                        *(setting[k] for k in ("loss", "kind", "alpha", "rho", "power"))
                    )
                    query = data["color"][np.r_[fr, qr]] if probe else data["color"][held_rows]
                    transforms = [(0.0, np.zeros(3))] if probe else [SETTINGS[0], *SETTINGS[9:17]]
                    biggest = 0.0
                    for dose, anchor in transforms:
                        xq = transformed(query, dose, anchor)
                        biggest = max(
                            biggest,
                            float(np.abs(predict(fitted, xq) - predict(expected, xq)).max()),
                        )
                        checks["qr_query_rows"] += len(query)
                    assert biggest <= 0.001, (role, fold, seed, setting, biggest)
                    maxima["qr_prediction"] = max(maxima["qr_prediction"], biggest)
                    checks[
                        "forced_positive_reconstructions" if probe else "selected_reconstructions"
                    ] += 1
                    reconstructions.append(
                        dict(
                            role=role,
                            fold=fold,
                            seed=seed,
                            loss=setting["loss"],
                            kind=setting["kind"],
                            alpha=setting["alpha"],
                            rho=setting["rho"],
                            power=setting["power"],
                            probe=probe,
                            max_lab_drift=biggest,
                        )
                    )
            geometry_records.append(
                dict(
                    role=role,
                    fold=fold,
                    rows=len(fr),
                    raw_width=g.rw,
                    projected_width=g.pw,
                    active_gate=g.beta is not None,
                )
            )
            if qr is not None:
                oof = nz(path / "oof.npz")
                np.testing.assert_array_equal(oof["row_indices"], qr)
                ix, cache = np.searchsorted(rows, qr), {}
                seen.extend(ix.tolist())
                for name, m in models.items():
                    pred = cached_prediction(m, data["color"][qr], cache)
                    delta = float(np.abs(pred - oof["pred__" + name]).max())
                    assert delta <= 2e-8
                    maxima["oof"] = max(maxima["oof"], delta)
                    checks["oof_rows"] += len(qr)
                    if name not in merged:
                        merged[name] = np.empty((len(rows), 3))
                    merged[name][ix] = pred
                    sizes[name] = max(sizes.get(name, 0), sum(v.nbytes for v in m.values()))
            checks["banks"] += 1
            checks["stored_readouts"] += len(models)
            print(f"AUDIT geometry/learning {role}/{sub}", flush=True)
        np.testing.assert_array_equal(sorted(seen), np.arange(len(rows)))

        def scores(name):
            return scoring(merged[name], data["target"][rows], data["patient"][rows])

        for loss in LOSSES:
            for kind in KINDS:
                candidates = []
                registered = selections["roles"][role][loss][kind]
                for setting, recorded in zip(options(kind), registered["candidates"], strict=True):
                    names = [ident(loss, seed, **setting) for seed in SEEDS]
                    ss = [scores(name) for name in names]
                    candidate = dict(
                        **setting,
                        clean=float(np.mean([s["clean"] for s in ss])),
                        p90=float(np.mean([s["p90"] for s in ss])),
                        numeric_bytes=max(sizes[n] for n in names),
                        seed_scores=ss,
                    )
                    numeric_equal(candidate, recorded)
                    candidates.append(candidate)
                    checks["candidate_scores"] += 1
                threshold = min(c["clean"] for c in candidates) + 1e-5
                chosen = sorted(
                    (c for c in candidates if c["clean"] <= threshold),
                    key=lambda c: (
                        c["numeric_bytes"],
                        c["clean"],
                        c["p90"],
                        -c["alpha"],
                        c["rho"],
                        c["power"],
                    ),
                )[0]
                numeric_equal(chosen, registered["selected"])
                checks["choices"] += 1
        for loss in LOSSES:
            ss = [scores(f"x_fixed_{loss}_s{seed}") for seed in SEEDS]
            close(
                dict(
                    clean=float(np.mean([s["clean"] for s in ss])),
                    p90=float(np.mean([s["p90"] for s in ss])),
                    seed_scores=ss,
                ),
                selections["references"][role][f"x_fixed_{loss}"],
            )
        close(scores("constant"), selections["references"][role]["constant"])
        checks["references"] += 3
        xx = [transformed(data["color"][held_rows], t, a) for t, a in SETTINGS]
        for rec in (r for r in result["records"] if r["role"] == role):
            mp, fp = (
                run / "selected" / role / f"{rec['name']}.npz",
                run / "evaluated" / role / f"{rec['name']}.npz",
            )
            assert sha(mp) == rec["model_sha256"] and sha(fp) == rec["prediction_sha256"]
            m, saved = nz(mp), nz(fp)
            exact(m, models[rec["model_id"]])
            np.testing.assert_array_equal(saved["row_indices"], held_rows)
            if rec["policy"] == "inner":
                family = rec["family"].split("_", 1)[1]
                chosen = selections["roles"][role][rec["loss"]][family]["selected"]
                assert rec["model_id"] == ident(
                    rec["loss"],
                    rec["seed"],
                    **{k: chosen[k] for k in ("kind", "alpha", "rho", "power")},
                )
            portable = Predictor(m)
            pred = np.array([[portable(v) for v in x] for x in xx])
            delta = float(np.abs(pred - saved["prediction"]).max())
            assert delta <= 2e-8
            maxima["standalone"] = max(maxima["standalone"], delta)
            d = float(np.abs(predict(m, xx[0]) - pred[0]).max())
            assert d <= 2e-8
            maxima["direct_identity"] = max(maxima["direct_identity"], d)
            gates = None if "gate_beta" not in m else np.array([gate_score(m, x) for x in xx])
            if gates is not None:
                np.testing.assert_allclose(gates, saved["scores"], atol=2e-8, rtol=0)
            summaries(
                rec,
                pred,
                data["target"][held_rows],
                data["patient"][held_rows],
                data["device"][held_rows],
                gates,
            )
            _, errors = error_summary(
                pred[0],
                data["target"][held_rows],
                data["patient"][held_rows],
                data["site"][held_rows],
            )
            person_errors[(role, rec["family"], rec["seed"])] = errors
            checks["final_query_rows"] += len(held_rows) * 33
            checks["final_models"] += 1
            checks["dose_summaries"] += 4
        print(f"AUDIT all33 transforms, actual NumPy consumer {role}", flush=True)
    assert maxima["width"] <= 2e-6
    expected = dict(
        banks=12,
        stored_readouts=2964,
        exact_A_controls=72,
        imported_X_controls=72,
        endpoint_aliases=216,
        covariance_svd=12,
        shared_bases=36,
        widths=24,
        oof_rows=419900,
        candidate_scores=258,
        choices=30,
        references=9,
        final_query_rows=1462758,
        final_models=111,
        dose_summaries=444,
        selected_reconstructions=90,
        forced_positive_reconstructions=18,
        new_coefficient_solutions=936,
        gram_decompositions=288,
        basis_preparations=72,
        exact_width_calculations=24,
        covariance_eigendecompositions=12,
        gate_fits=4,
        auxiliary_baseline_solves=36,
        auxiliary_theta_solves=36,
    )
    for k, v in expected.items():
        assert checks[k] == v, (k, checks[k], v)
    paired = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        people = np.unique(data["patient"][held])
        groups = np.array(
            [np.unique(data["device"][held & (data["patient"] == p)])[0] for p in people]
        )
        rng = np.random.default_rng(716031)
        draws = np.column_stack(
            [
                rng.choice(np.flatnonzero(groups == g), (2000, (groups == g).sum()))
                for g in np.unique(groups)
            ]
        )
        for loss in LOSSES:
            anchor = np.mean([person_errors[(role, loss + "_raw", s)] for s in SEEDS], axis=0)
            for kind in KINDS[1:]:
                e = np.mean([person_errors[(role, loss + "_" + kind, s)] for s in SEEDS], axis=0)
                diff = e - anchor
                paired.append(
                    dict(
                        role=role,
                        family=loss + "_" + kind,
                        mean_difference=float(diff.mean()),
                        people=len(diff),
                        people_improve=int((diff < 0).sum()),
                        conditional_bootstrap_95=np.quantile(
                            diff[draws].mean(1), [0.025, 0.975]
                        ).tolist(),
                        draws=2000,
                        fixed_prediction_only=True,
                        independent_validation=False,
                    )
                )
    deps = [
        "scripts/chromaseed_hybrid_reference.py",
        "scripts/chromaseed_hybrid_numpy.py",
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_affine_reference.py",
        "scripts/chromaseed_gate_stability_audit.py",
        "scripts/chromaseed_gated_audit.py",
        "scripts/chromaseed_kernel_audit.py",
        "scripts/chromaseed_perceptual_audit.py",
        "scripts/chromaseed_perceptual_reference.py",
        "scripts/chromaseed_projection_audit.py",
        "scripts/chromaseed_projection_reference.py",
        "scripts/chromaseed_refine_audit.py",
    ]
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
            geometry=geometry_records,
            reconstructions=reconstructions,
            paired=paired,
            wall_seconds=time.perf_counter() - start,
        ),
    )
    print("H independent audit PASS " + str(maxima), flush=True)


if __name__ == "__main__":
    main()
