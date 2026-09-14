"""Separate algebra/row/selection audit of the frozen ChromaSeed-K run."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json
from skin_local_search_train import predict as legacy_predict


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def key_for(family, rank, seed, wi, ai):
    return f"{family}_k{rank:03d}_s{seed}_w{wi}_a{ai}"


def unpack(arrays, key):
    fields = ("x_mean", "x_std", "y_mean", "y_std", "width", "centers", "coefficient")
    return {f: arrays[f"{key}__{f}"] for f in fields}


def norm(model, x):
    return ((x.astype(np.float32) - model["x_mean"]) / model["x_std"]).astype(np.float64)


def direct_kernel(a, b, width):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    result = np.empty((len(a), len(b)), np.float64)
    for start in range(0, len(a), 24):
        difference = a[start:start + 24, None] - b[None]
        result[start:start + 24] = np.exp(-np.mean(difference * difference, axis=-1) / (2. * float(width) ** 2))
    return result


def independent_predict(model, x):
    return direct_kernel(norm(model, x), model["centers"], model["width"]) @ model["coefficient"].astype(np.float64) * model["y_std"] + model["y_mean"]


def svd_refit(kernel, y, w, config, teachers):
    alpha = config["alpha"]
    if alpha not in teachers:
        teachers[alpha] = np.linalg.solve(kernel + np.diag(alpha / w), y).astype(np.float32)
    if config["family"] == "exact":
        return teachers[alpha]
    idx = config["fit_center_indices"]
    sub = kernel[np.ix_(idx, idx)]
    u, singular, _ = np.linalg.svd(sub, full_matrices=False)
    keep = singular > 1e-8 * singular.max()
    basis = u[:, keep] / np.sqrt(singular[keep])
    if config["family"] == "project_rpchol":
        beta = basis @ (basis.T @ (kernel[idx] @ teachers[alpha].astype(np.float64)))
    else:
        design = kernel[:, idx] @ basis
        gram = design.T @ (w[:, None] * design) + alpha * np.eye(keep.sum())
        right = design.T @ (w[:, None] * y)
        beta = basis @ np.linalg.solve(gram, right)
    return beta.astype(np.float32)


def summarize(predictions, data, idx):
    rows = [error_summary(p, data["target"][idx], data["patient"][idx], data["site"][idx])[0] for p in predictions]
    return float(np.mean([r["person_mean"] for r in rows])), float(np.mean([r["p90"] for r in rows]))


def policy(trace, bounds, ranks, tolerance):
    outputs, counts, met = [], [], []
    for i, row in enumerate(trace):
        level = len(row) - 1
        if tolerance > 0:
            level = next((j for j in range(len(row)) if bounds[i, j] <= tolerance), level)
        outputs.append(row[level])
        counts.append(ranks[i, level] if ranks.ndim == 2 else ranks[level])
        met.append(tolerance > 0 and bounds[i, level] <= tolerance)
    return np.stack(outputs), np.array(counts), np.array(met)


def adaptive_trace(payload, x):
    features = direct_kernel(norm(payload, x), payload["centers"], payload["width"])
    outputs, bounds = [], []
    offset = 0
    for i, rank in enumerate(payload["ranks"]):
        rank = int(rank)
        coefficient = payload["coefficients"][offset:offset + rank].astype(np.float64)
        residuals = payload["residual_values"][offset:offset + rank]
        q = payload["norm_sq"][i]
        k = features[:, :rank]
        closest = np.argmax(k, axis=1)
        correlation = np.clip(k[np.arange(len(k)), closest], 0., 1.)[:, None]
        residual = residuals[closest]
        component = np.abs(correlation * residual) + np.sqrt(np.maximum(q - residual ** 2, 0.)) * np.sqrt(np.maximum(1 - correlation ** 2, 0.))
        component += 1e-9 * (1 + np.sqrt(q))
        bounds.append(np.sqrt(np.sum((component * payload["y_std"]) ** 2, axis=1)))
        outputs.append((k @ coefficient) * payload["y_std"] + payload["y_mean"])
        offset += rank
    return np.stack(outputs, axis=1), np.stack(bounds, axis=1)


def main_audit(run, cache, legacy, output):
    if cache.name != "train.npz" or sha(cache) != CACHE_HASH:
        raise ValueError("unexpected data")
    lock, selected, results = js(run / "source_lock.json"), js(run / "selections.json"), js(run / "results.json")
    lock_hash = sha(run / "source_lock.json")
    for path, digest in lock["sources"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"frozen source changed: {path}")
    for path, digest in lock["legacy_bindings"].items():
        if sha(legacy / path) != digest:
            raise ValueError("legacy binding changed")
    if results["source_lock_sha256"] != lock_hash or selected["source_lock_sha256"] != lock_hash or results["selection_sha256"] != sha(run / "selections.json"):
        raise ValueError("selection/results binding mismatch")
    with np.load(cache, allow_pickle=False) as z:
        data = {key: z[key] for key in ("color", "target", "patient", "site", "device")}
    counters = {"bank_file_hashes": 0, "bank_model_normalizers_and_centers": 0, "oof_model_probes": 0, "oof_probe_rows": 0,
                "primary_selection_decisions": 0, "adaptive_selection_decisions": 0, "blend_selection_decisions": 0,
                "primary_svd_refits": 0, "final_model_hashes": 0, "final_prediction_rows": 0}
    maximum_prediction_drift, maximum_refit_drift, maximum_bound_violation = 0., 0., -np.inf
    paired = []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        fit_idx, held_idx = np.flatnonzero(fit), np.flatnonzero(held)
        folds = folds_for(data["patient"][fit], data["device"][fit])
        oof_parts, rank_parts, oof_rows = {}, {}, []
        final_bank, final_receipt = None, None
        for fold in (0, 1, 2, None):
            directory = run / ("final" if fold is None else "inner") / role / ("bank" if fold is None else f"fold{fold}")
            receipt = js(directory / "receipt.json")
            if receipt["source_lock_sha256"] != lock_hash:
                raise ValueError("bank source mismatch")
            for name, digest in receipt["files"].items():
                if sha(directory / name) != digest:
                    raise ValueError("bank file changed")
                counters["bank_file_hashes"] += 1
            bank = nz(directory / "models.npz")
            train_idx = fit_idx if fold is None else fit_idx[folds != fold]
            x, y = data["color"][train_idx].astype(np.float64), data["target"][train_idx].astype(np.float64)
            expected = {"x_mean": x.mean(0).astype(np.float32), "x_std": np.maximum(x.std(0), 1e-6).astype(np.float32),
                        "y_mean": y.mean(0).astype(np.float32), "y_std": np.maximum(y.std(0), 1e-6).astype(np.float32)}
            canonical_x = norm(expected, x)
            for key, config in receipt["models"].items():
                model = unpack(bank, key)
                for field, value in expected.items():
                    np.testing.assert_array_equal(model[field], value)
                indices = np.array(config["fit_center_indices"])
                assert len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < len(train_idx)))
                np.testing.assert_array_equal(model["centers"], canonical_x[indices].astype(np.float32))
                counters["bank_model_normalizers_and_centers"] += 1
            if fold is None:
                if receipt["selection_sha256"] != sha(run / "selections.json"):
                    raise ValueError("final bank fitted for different choices")
                final_bank, final_receipt = bank, receipt
                continue
            validation_idx = fit_idx[folds == fold]
            if set(data["patient"][train_idx]) & set(data["patient"][validation_idx]):
                raise ValueError("person leakage")
            predictions = nz(directory / "oof.npz")
            np.testing.assert_array_equal(predictions["row_indices"], validation_idx)
            oof_rows.append(validation_idx)
            probes = np.unique(np.linspace(0, len(validation_idx) - 1, 3).astype(int))
            for key, config in receipt["models"].items():
                model = unpack(bank, key)
                predicted = independent_predict(model, data["color"][validation_idx[probes]])
                drift = float(np.abs(predicted - predictions[f"pred__{key}"][probes]).max())
                if drift > 1e-5:
                    raise ValueError(f"OOF independent kernel drift {drift}")
                maximum_prediction_drift = max(maximum_prediction_drift, drift)
                counters["oof_model_probes"] += 1
                counters["oof_probe_rows"] += len(probes)
                rank_parts.setdefault(key, []).append(np.full(len(validation_idx), config["actual_centers"]))
            for key, value in predictions.items():
                if key != "row_indices":
                    oof_parts.setdefault(key, []).append(value)
        rows = np.concatenate(oof_rows)
        order = np.argsort(rows)
        np.testing.assert_array_equal(rows[order], fit_idx)
        oof = {key: np.concatenate(value)[order] for key, value in oof_parts.items()}
        ranks = {key: np.concatenate(value)[order] for key, value in rank_parts.items()}
        choice = selected["roles"][role]
        for family, rank_choices in choice["primary"].items():
            seeds = [17] if family in ("exact", "nys_pivot") else [17, 29, 43]
            for rank, saved in rank_choices.items():
                scored = []
                for wi, factor in enumerate(lock["width_factors"]):
                    for ai, alpha in enumerate(lock["alphas"]):
                        values = [oof[f"pred__{key_for(family, int(rank), seed, wi, ai)}"] for seed in seeds]
                        mean, p90 = summarize(values, data, fit_idx)
                        stored = next(c for c in saved["candidates"] if c["width_index"] == wi and c["alpha_index"] == ai)
                        np.testing.assert_allclose([mean, p90], [stored["person_mean"], stored["p90"]], atol=1e-10, rtol=0)
                        scored.append((mean, alpha, factor, wi, ai))
                winner = min(scored)
                assert (winner[3], winner[4]) == (saved["width_index"], saved["alpha_index"])
                counters["primary_selection_decisions"] += 1
        adaptive = choice["adaptive"]
        wi, ai = adaptive["width_index"], adaptive["alpha_index"]
        scored = []
        for tolerance in lock["tolerances"]:
            values, used = [], []
            for seed in (17, 29, 43):
                keys = [key_for("project_rpchol", rank, seed, wi, ai) for rank in lock["ranks"]]
                trace = np.stack([oof[f"pred__{key}"] for key in keys], axis=1)
                bound = np.stack([oof[f"bound__{key}"] for key in keys], axis=1)
                actual_ranks = np.stack([ranks[key] for key in keys], axis=1)
                prediction, count, _ = policy(trace, bound, actual_ranks, tolerance)
                values.append(prediction)
                used.append(count.mean())
            mean, p90 = summarize(values, data, fit_idx)
            scored.append((float(np.mean(used)), tolerance, mean, p90))
        fixed = scored[0]
        acceptable = [r for r in scored if r[2] <= fixed[2] + .02 and r[3] <= fixed[3] + .1]
        assert min(acceptable)[:2] == (adaptive["selected"]["mean_centers"], adaptive["tolerance"])
        counters["adaptive_selection_decisions"] += 1
        guided_oof = {}
        old_alpha = js(legacy / role / "selection.json")["guided_rbf"]["parameter"]
        ci = lock["alphas"].index(old_alpha)
        for seed in (17, 29, 43):
            values = []
            for fold, row in enumerate(oof_rows):
                model = nz(legacy / role / "inner" / f"guided_rbf_c{ci}_s{seed}_f{fold}.npz")
                values.append(legacy_predict(model, data["color"][row]))
            guided_oof[seed] = np.concatenate(values)[order]
        for blend in choice["blends"].values():
            scores = []
            for rho in lock["blend_weights"]:
                values = [(1 - rho) * oof[f"pred__{key_for(blend['family'], blend['rank'], seed, blend['width_index'], blend['alpha_index'])}"] + rho * guided_oof[seed] for seed in (17, 29, 43)]
                mean, _ = summarize(values, data, fit_idx)
                scores.append((mean, rho))
            assert min(scores)[1] == blend["rho"]
            counters["blend_selection_decisions"] += 1
        # Independent direct solves/SVD refits for all selected primary models.
        example = unpack(final_bank, key_for("exact", 0, 17, 1, 0))
        train_x = norm(example, data["color"][fit_idx])
        train_y = (data["target"][fit_idx] - example["y_mean"]) / example["y_std"]
        w = weights_for(data["patient"][fit_idx], data["site"][fit_idx])
        kernels, teachers = {}, {}
        per_person = {}
        for record in [r for r in results["records"] if r["role"] == role and r["family"] != "exact_width1_reference"]:
            family, seed, rank = record["family"], record["seed"], record["rank"]
            filename = f"{family}_s{seed}.npz" if family == "adaptive_project" or family.startswith("blend_") else f"{family}_k{rank:03d}_s{seed}.npz"
            model_path = run / "selected" / role / filename
            if sha(model_path) != record["model_sha256"]:
                raise ValueError("selected model file changed")
            counters["final_model_hashes"] += 1
            model, evaluation = nz(model_path), nz(run / "evaluated" / role / filename)
            np.testing.assert_array_equal(evaluation["row_indices"], held_idx)
            if family.startswith("blend_"):
                rho = float(model["rho"])
                prediction = 0.
                if rho < 1:
                    prediction = (1 - rho) * independent_predict({k.removeprefix("base__"): v for k, v in model.items() if k.startswith("base__")}, data["color"][held_idx])
                if rho > 0:
                    prediction = prediction + rho * legacy_predict({k.removeprefix("guided__"): v for k, v in model.items() if k.startswith("guided__")}, data["color"][held_idx])
            elif family == "adaptive_project":
                trace, bounds = adaptive_trace(model, data["color"][held_idx])
                prediction, used, met = policy(trace, bounds, model["ranks"], float(model["tolerance"]))
                np.testing.assert_array_equal(used, evaluation["used_centers"])
                np.testing.assert_array_equal(met, evaluation["tolerance_met"])
                np.testing.assert_allclose(bounds, evaluation["bounds"], rtol=1e-8, atol=1e-7)
                teacher = unpack(final_bank, key_for("exact", 0, 17, adaptive["width_index"], adaptive["alpha_index"]))
                teacher_prediction = independent_predict(teacher, data["color"][held_idx])
                violation = float(np.max(np.linalg.norm(trace - teacher_prediction[:, None], axis=-1) - bounds))
                assert violation <= 1e-6
                maximum_bound_violation = max(maximum_bound_violation, violation)
            else:
                key = key_for(family, rank, seed, record["width_index"], record["alpha_index"])
                config = final_receipt["models"][key]
                wi = record["width_index"]
                if wi not in kernels:
                    kernels[wi] = direct_kernel(train_x, train_x, model["width"])
                    teachers[wi] = {}
                beta = svd_refit(kernels[wi], train_y, w, config, teachers[wi])
                reconstructed = {**model, "coefficient": beta}
                probes = np.unique(np.linspace(0, len(held_idx) - 1, 7).astype(int))
                refit_prediction = independent_predict(reconstructed, data["color"][held_idx[probes]])
                stored_prediction = independent_predict(model, data["color"][held_idx[probes]])
                difference = float(np.abs(refit_prediction - stored_prediction).max())
                if difference > .002:
                    raise ValueError(f"independent SVD refit drift {difference}: {role}/{family}/{rank}/{seed}")
                maximum_refit_drift = max(maximum_refit_drift, difference)
                counters["primary_svd_refits"] += 1
                prediction = independent_predict(model, data["color"][held_idx])
            difference = float(np.abs(prediction - evaluation["prediction"]).max())
            if difference > 1e-5:
                raise ValueError("full final prediction drift")
            maximum_prediction_drift = max(maximum_prediction_drift, difference)
            measured, person_errors = error_summary(evaluation["prediction"], data["target"][held_idx], data["patient"][held_idx], data["site"][held_idx])
            for metric in ("person_mean", "image_mean", "site_person_mean", "p90", "median", "gt5", "gt10"):
                np.testing.assert_allclose(measured[metric], record["metrics"][metric], atol=1e-10, rtol=0)
            per_person.setdefault((family, rank), []).append(person_errors)
            counters["final_prediction_rows"] += len(held_idx)
        reference_errors = np.mean(per_person[("exact", 0)], axis=0)
        rng = np.random.default_rng(933017)
        sample = rng.integers(len(reference_errors), size=(20000, len(reference_errors)))
        for (family, rank), values in per_person.items():
            if family == "exact":
                continue
            difference = np.mean(values, axis=0) - reference_errors
            interval = np.quantile(difference[sample].mean(1), [.025, .975])
            paired.append({"role": role, "family": family, "rank": rank, "mean_delta_vs_exact": float(difference.mean()),
                           "descriptive_person_bootstrap_95": interval.tolist(), "n_people": len(difference), "per_person_differences": difference.tolist()})
        print(f"AUDITED {role}", flush=True)
    result = {"passed": True, "source_lock_sha256": lock_hash, "selection_sha256": sha(run / "selections.json"),
              "audit_source_sha256": sha(Path(__file__)), "checks": counters, "maximum_independent_prediction_component_drift": maximum_prediction_drift,
              "maximum_independent_svd_refit_component_drift": maximum_refit_drift, "maximum_teacher_bound_violation": maximum_bound_violation,
              "paired": paired, "evidence": "implementation-independent recomputation on reused exploratory people, not external replication or fresh validation",
              "legacy_predictor": "unchanged previously audited canonical predictor reused for the old RBF component"}
    write_json(output / "audit.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_audit(args.run, args.cache, args.legacy_run, args.output)
