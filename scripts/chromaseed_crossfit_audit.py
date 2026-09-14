"""C whole-person routing, independent teacher/head refits and actual consumer audit."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact, summaries, validate_normalizer
from chromaseed_crossfit_reference import head, teacher
from chromaseed_gate_stability_audit import ANCHORS, DOSES, close, transformed
from chromaseed_gate_stability_audit import score as gate_score
from chromaseed_gated_audit import model_from
from chromaseed_hybrid_numpy import Predictor
from chromaseed_hybrid_reference import Basis, Geometry, predict
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/runs/chromaseed_hybrid_v1"
SEEDS, LOSSES = (17, 29, 43), ("norm", "perceptual")
TRANSFORMS = [(0.0, np.zeros(3))] + [(t, a) for t in DOSES for a in ANCHORS]


def main():
    parser = argparse.ArgumentParser()
    for k in ("run", "cache", "output"):
        parser.add_argument("--" + k, type=Path, required=True)
    args = parser.parse_args()
    run, out, start = args.run, args.output, time.perf_counter()
    lock, fixed, result = (
        js(run / p) for p in ("source_lock.json", "frozen_settings.json", "results.json")
    )
    source = sha(run / "source_lock.json")
    assert source == fixed["source_lock_sha256"] == result["source_lock_sha256"]
    assert result["settings_sha256"] == sha(run / "frozen_settings.json")
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH == lock["cache_sha256"]
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    assert len(lock["sources"]) == 68 and len(lock["input_sha256"]) == 229
    previous = js(PARENT / "selections.json")
    for role, losses in fixed["settings"].items():
        for loss, kinds in losses.items():
            for kind, s in kinds.items():
                original = previous["roles"][role][loss][kind]["selected"]
                assert s == {k: original[k] for k in ("kind", "alpha", "rho", "power")}
    assert fixed["new_C_hyperparameter_selection"] is False
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("color", "target", "patient", "site", "device")}
    checks = dict(
        banks=0,
        teacher_subsets=0,
        teacher_models=0,
        teacher_prediction_rows=0,
        teacher_portable_rows=0,
        routed_rows=0,
        matched_row_memberships=0,
        student_raw_refits=0,
        head_refits=0,
        head_refit_query_rows=0,
        exact_H_models=0,
        exact_H_prediction_rows=0,
        final_models=0,
        final_query_rows=0,
        dose_summaries=0,
        supervision_diagnostics=0,
        analytic_solutions=0,
        gram_decompositions=0,
        basis_preparations=0,
        widths=0,
        covariances=0,
        gates=0,
        auxiliary_baseline_solves=0,
        auxiliary_theta_solves=0,
    )
    maxima = dict(
        teacher_direct=0.0,
        teacher_portable=0.0,
        teacher_qr=0.0,
        head_qr=0.0,
        final_portable=0.0,
        final_direct=0.0,
        normal_residual=0.0,
        width=0.0,
    )
    fits, pe, diagnostic_rows = [], {}, []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        rows, queries = np.flatnonzero(fit), np.flatnonzero(held)
        directory = run / "banks" / role
        rec, bank, teacher_bank, tables = (
            js(directory / "receipt.json"),
            nz(directory / "models.npz"),
            nz(directory / "teachers.npz"),
            nz(directory / "tables.npz"),
        )
        assert rec["source_lock_sha256"] == source and rec["settings_sha256"] == sha(
            run / "frozen_settings.json"
        )
        for p, h in rec["files"].items():
            assert sha(directory / p) == h
        np.testing.assert_array_equal(tables["row_indices"], rows)
        xx, yy, pp, ss, cc = (
            data[k][rows] for k in ("color", "target", "patient", "site", "device")
        )
        people = np.unique(pp)
        np.testing.assert_array_equal(rec["people"], people)
        assert not set(pp) & set(data["patient"][queries])
        own = np.array([np.flatnonzero(people == p)[0] for p in pp])
        matched = np.empty(len(pp), np.int64)
        for p in people:
            camera = np.unique(cc[pp == p])
            assert len(camera) == 1
            group = np.unique(pp[cc == camera[0]])
            assert len(group) >= 2
            nextp = group[(np.flatnonzero(group == p)[0] + 1) % len(group)]
            matched[pp == p] = np.flatnonzero(people == nextp)[0]
        np.testing.assert_array_equal(own, tables["own_teacher"])
        np.testing.assert_array_equal(matched, tables["matched_teacher"])
        assert np.all(own != matched)
        models = {case["name"]: model_from(bank, case["name"]) for case in rec["cases"]}
        assert len(models) == 61 and len(bank) == sum(len(m) for m in models.values())
        assert len(rec["teacher_names"]) == 6 * len(people)
        independent_tables = {
            f"{loss}_s{seed}": np.empty((len(people), len(rows), 3))
            for loss in LOSSES
            for seed in SEEDS
        }
        for j, p in enumerate(people):
            fr, excluded = np.flatnonzero(pp != p), np.flatnonzero(pp == p)
            sub = rec["subsets"][j]
            np.testing.assert_array_equal(sub["fit_local_rows"], fr)
            np.testing.assert_array_equal(sub["excluded_local_rows"], excluded)
            np.testing.assert_array_equal(sub["fit_person_values"], np.unique(pp[fr]))
            assert sub["n_fit_people"] == len(people) - 1 and sub["n_fit_rows"] == len(fr)
            assert not set(pp[fr]) & {p}
            assert sub["active_gate"] == (len(np.unique(cc[fr])) > 1)
            for op in sub["operations"]:
                checks["gram_decompositions"] += 1
                for s in op["solutions"]:
                    assert s["normal_residual"] <= 1e-8 and s["objective_minus_zero"] <= 1e-8
                    maxima["normal_residual"] = max(maxima["normal_residual"], s["normal_residual"])
                    checks["analytic_solutions"] += 1
            for seed in SEEDS:
                basis = next(b for b in sub["bases"] if b["seed"] == seed)
                for loss in LOSSES:
                    name = f"exclude{j}_{loss}_s{seed}"
                    m = model_from(teacher_bank, name)
                    validate_normalizer(m, xx[fr], yy[fr])
                    direct = predict(m, xx)
                    saved = tables[f"native__{loss}_s{seed}"][j]
                    d = float(np.abs(direct - saved).max())
                    assert d <= 2e-8
                    maxima["teacher_direct"] = max(maxima["teacher_direct"], d)
                    call = Predictor(m)
                    d = float(np.abs(np.array([call(v) for v in xx]) - saved).max())
                    assert d <= 2e-8
                    maxima["teacher_portable"] = max(maxima["teacher_portable"], d)
                    independent, info = teacher(
                        m, xx[fr], yy[fr], pp[fr], ss[fr], cc[fr], loss, seed
                    )
                    np.testing.assert_array_equal(
                        fr[info["fit_center_indices"]], basis["center_local_rows"]
                    )
                    assert info["effective_rank"] == basis["effective_rank"]
                    maxima["width"] = max(maxima["width"], info["width_drift"])
                    pred = predict(independent, xx)
                    d = float(np.abs(pred - saved).max())
                    assert d <= 0.001
                    maxima["teacher_qr"] = max(maxima["teacher_qr"], d)
                    independent_tables[f"{loss}_s{seed}"][j] = pred
                    checks["teacher_models"] += 1
                    checks["teacher_prediction_rows"] += len(rows)
                    checks["teacher_portable_rows"] += len(rows)
                    fits.append(
                        dict(
                            role=role,
                            excluded_index=j,
                            seed=seed,
                            loss=loss,
                            kind="teacher",
                            max_lab_drift=d,
                        )
                    )
            checks["teacher_subsets"] += 1
            checks["basis_preparations"] += 3
            checks["widths"] += 1
            checks["gates"] += int(sub["active_gate"])
            checks["auxiliary_baseline_solves"] += 3
            checks["auxiliary_theta_solves"] += 3
        for i in range(len(rows)):
            a, b = rec["subsets"][own[i]], rec["subsets"][matched[i]]
            assert i not in a["fit_local_rows"] and i in b["fit_local_rows"]
            for camera in np.unique(cc):
                assert len(
                    np.unique(
                        pp[np.array(a["fit_local_rows"])][
                            cc[np.array(a["fit_local_rows"])] == camera
                        ]
                    )
                ) == len(
                    np.unique(
                        pp[np.array(b["fit_local_rows"])][
                            cc[np.array(b["fit_local_rows"])] == camera
                        ]
                    )
                )
            checks["matched_row_memberships"] += 1
        print(
            f"AUDIT C {role}:all{len(people) * 6} teachers independently reconstructed", flush=True
        )
        routed = {}
        for loss in LOSSES:
            for seed in SEEDS:
                name = f"{loss}_s{seed}"
                for arm, index in (("out_person", own), ("in_matched", matched)):
                    stored = tables["native__" + name][index, np.arange(len(rows))]
                    np.testing.assert_array_equal(stored, tables[f"{arm}__{name}"])
                    routed[(arm, loss, seed)] = independent_tables[name][
                        index, np.arange(len(rows))
                    ]
                    checks["routed_rows"] += len(rows)
        geometry = Geometry(xx, yy, pp, ss, cc)
        for seed in SEEDS:
            b = Basis(geometry, seed)
            original_basis = next(v for v in rec["student_bases"] if v["seed"] == seed)
            np.testing.assert_array_equal(b.ids, original_basis["center_local_rows"])
            assert (
                b.rwhite.shape[1] == original_basis["raw_rank"]
                and b.pwhite.shape[1] == original_basis["projected_rank"]
            )
            for loss in LOSSES:
                full = models[f"h_{loss}_raw_s{seed}"]
                validate_normalizer(full, xx, yy)
                np.testing.assert_allclose(
                    predict(full, xx), tables[f"full__{loss}_s{seed}"], rtol=0, atol=2e-8
                )
                assert (
                    np.max(np.abs(predict(b.raw[loss], xx) - tables[f"full__{loss}_s{seed}"]))
                    <= 0.001
                )
                checks["student_raw_refits"] += 1
                for kind, setting in fixed["settings"][role][loss].items():
                    for arm in ("in_matched", "out_person"):
                        m = models[f"{arm}_{loss}_{kind}_s{seed}"]
                        reference = head(b, routed[(arm, loss, seed)], loss, setting)
                        biggest = 0.0
                        for t, anchor in [TRANSFORMS[0], *TRANSFORMS[9:17]]:
                            q = transformed(data["color"][queries], t, anchor)
                            biggest = max(
                                biggest, float(np.abs(predict(reference, q) - predict(m, q)).max())
                            )
                            checks["head_refit_query_rows"] += len(queries)
                        assert biggest <= 0.001
                        maxima["head_qr"] = max(maxima["head_qr"], biggest)
                        fits.append(
                            dict(
                                role=role,
                                seed=seed,
                                loss=loss,
                                kind=kind,
                                arm=arm,
                                max_lab_drift=biggest,
                            )
                        )
                        checks["head_refits"] += 1
        checks["basis_preparations"] += 6
        checks["widths"] += 2
        checks["covariances"] += 1
        checks["gates"] += rec["student_gate_fits"]
        checks["auxiliary_baseline_solves"] += 3
        checks["auxiliary_theta_solves"] += 3
        for op in rec["student_operations"]:
            checks["gram_decompositions"] += 1
            for s in op["solutions"]:
                assert s["normal_residual"] <= 1e-8 and s["objective_minus_zero"] <= 1e-8
                maxima["normal_residual"] = max(maxima["normal_residual"], s["normal_residual"])
                checks["analytic_solutions"] += 1
        for d in rec["supervision_diagnostics"]:
            table = tables[f"{d['arm']}__{d['loss']}_s{d['seed']}"]
            metrics, _ = error_summary(table, yy, pp, ss)
            close(metrics, d["metrics"])
            close((yy - table).mean(0).tolist(), d["mean_signed_native_residual"])
            close(float(np.sqrt(np.mean((yy - table) ** 2))), d["native_residual_rms"])
            close(
                float(np.sqrt(np.mean((table - tables[f"full__{d['loss']}_s{d['seed']}"]) ** 2))),
                d["full_backbone_discrepancy_rms"],
            )
            diagnostic_rows.append(dict(role=role, **d))
            checks["supervision_diagnostics"] += 1
        transformed_queries = [transformed(data["color"][queries], t, a) for t, a in TRANSFORMS]
        for rec in (r for r in result["records"] if r["role"] == role):
            mp, fp = (
                run / "selected" / role / f"{rec['name']}.npz",
                run / "evaluated" / role / f"{rec['name']}.npz",
            )
            assert sha(mp) == rec["model_sha256"] and sha(fp) == rec["prediction_sha256"]
            m, saved = nz(mp), nz(fp)
            exact(m, models[rec["name"]])
            np.testing.assert_array_equal(saved["row_indices"], queries)
            assert rec["numeric_bytes"] == sum(v.nbytes for v in m.values())
            if rec["origin"] == "H":
                exact(m, nz(PARENT / "selected" / role / f"{rec['parent_name']}.npz"))
                np.testing.assert_array_equal(
                    saved["prediction"],
                    nz(PARENT / "evaluated" / role / f"{rec['parent_name']}.npz")["prediction"],
                )
                checks["exact_H_models"] += 1
                checks["exact_H_prediction_rows"] += len(queries) * 33
            else:
                s = fixed["settings"][role][rec["loss"]][rec["kind"]]
                assert s == {k: rec[k] for k in s}
            consumer = Predictor(m)
            pred = np.array([[consumer(v) for v in q] for q in transformed_queries])
            d = float(np.abs(pred - saved["prediction"]).max())
            assert d <= 2e-8
            maxima["final_portable"] = max(maxima["final_portable"], d)
            d = float(np.abs(predict(m, transformed_queries[0]) - pred[0]).max())
            assert d <= 2e-8
            maxima["final_direct"] = max(maxima["final_direct"], d)
            gates = (
                None
                if "gate_beta" not in m
                else np.array([gate_score(m, q) for q in transformed_queries])
            )
            if gates is not None:
                np.testing.assert_allclose(gates, saved["scores"], rtol=0, atol=2e-8)
            summaries(
                rec,
                pred,
                data["target"][queries],
                data["patient"][queries],
                data["device"][queries],
                gates,
            )
            metric, errors = error_summary(
                pred[0], data["target"][queries], data["patient"][queries], data["site"][queries]
            )
            close(metric, rec["metrics"])
            pe[(role, rec["family"], rec["seed"])] = errors
            checks["final_models"] += 1
            checks["final_query_rows"] += len(queries) * 33
            checks["dose_summaries"] += 4
        checks["banks"] += 1
        print(f"AUDIT C {role}:all61 actual consumers,33 transforms,matched H controls", flush=True)
    expected = dict(
        banks=3,
        teacher_subsets=42,
        teacher_models=252,
        teacher_prediction_rows=156504,
        teacher_portable_rows=156504,
        routed_rows=20400,
        matched_row_memberships=1700,
        student_raw_refits=18,
        head_refits=72,
        head_refit_query_rows=258768,
        exact_H_models=111,
        exact_H_prediction_rows=1462758,
        final_models=183,
        final_query_rows=2411574,
        dose_summaries=732,
        supervision_diagnostics=54,
        analytic_solutions=342,
        gram_decompositions=207,
        basis_preparations=144,
        widths=48,
        covariances=3,
        gates=19,
        auxiliary_baseline_solves=135,
        auxiliary_theta_solves=135,
    )
    assert checks == expected, checks
    assert maxima["width"] == 0
    paired = []
    for role, (_, held) in roles(data["patient"], data["device"]).items():
        people = np.unique(data["patient"][held])
        groups = np.array(
            [np.unique(data["device"][held & (data["patient"] == p)])[0] for p in people]
        )
        rng = np.random.default_rng(881031)
        draws = np.column_stack(
            [
                rng.choice(np.flatnonzero(groups == g), (2000, (groups == g).sum()))
                for g in np.unique(groups)
            ]
        )
        comparisons = []
        for loss in LOSSES:
            for kind in ("uniform", "support"):
                for arm in ("in_matched", "out_person"):
                    comparisons += [
                        (f"{arm}_{loss}_{kind}", f"h_{loss}_{kind}"),
                        (f"{arm}_{loss}_{kind}", f"h_{loss}_raw"),
                    ]
                comparisons.append((f"out_person_{loss}_{kind}", f"in_matched_{loss}_{kind}"))
        for family, control in comparisons:
            diff = np.mean([pe[(role, family, s)] - pe[(role, control, s)] for s in SEEDS], axis=0)
            paired.append(
                dict(
                    role=role,
                    family=family,
                    control=control,
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
    assert len(paired) == 60
    deps = (
        "scripts/chromaseed_crossfit_reference.py",
        "scripts/chromaseed_hybrid_reference.py",
        "scripts/chromaseed_projection_reference.py",
        "scripts/chromaseed_hybrid_numpy.py",
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_gate_stability_audit.py",
        "scripts/chromaseed_gated_audit.py",
        "scripts/chromaseed_refine_audit.py",
        "scripts/chromaseed_kernel_audit.py",
    )
    write_json(
        out / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=source,
            settings_sha256=sha(run / "frozen_settings.json"),
            results_sha256=sha(run / "results.json"),
            audit_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in deps},
            checks=checks,
            maxima=maxima,
            reconstructions=fits,
            supervision_diagnostics=diagnostic_rows,
            paired=paired,
            wall_seconds=time.perf_counter() - start,
        ),
    )
    print("C independent audit PASS " + str(maxima), flush=True)


if __name__ == "__main__":
    main()
