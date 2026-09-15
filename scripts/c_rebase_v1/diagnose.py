"""C_REBASE_V1 TRAIN OOF error-floor audit. No fitting, model selection or downloads.

Only complete OOF artifacts and a diagnostic contract frozen before OOF may be
analyzed. Outputs contain pseudonyms, never original image/patient/site IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("c_oof_diagnostic_helpers", ROOT / "scripts/error_floor/c_oof_audit.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)
from luma_skin_vision.color import delta_e00


EXPERIMENT = "C_REBASE_V1"
DIAGNOSTIC = "LEAKY_DIAGNOSTIC_ONLY"


def load_json(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def bootstrap_metrics(error, people, *, seed, draws):
    """Patient-cluster bootstrap retaining original image-weighted estimands."""
    unique = np.unique(people)
    indices = [np.flatnonzero(people == p) for p in unique]
    rng = np.random.default_rng(seed)
    values = np.empty((draws, 3))
    for b in range(draws):
        take = np.concatenate([indices[k] for k in rng.integers(0, len(unique), len(unique))])
        e = error[take]
        values[b] = [e.mean(), np.median(e), np.quantile(e, .95)]
    return {"unit": "patient", "draws": draws, "seed": seed, "method": "percentile_95_interval",
            "intervals": dict(zip(("mean", "median", "p95"), np.quantile(values, [.025, .975], axis=0).T.tolist())),
            "interpretation": "Fixed-recipe OOF descriptive uncertainty; not an independent confirmatory experiment."}


def decomposition(pred, target, sites, people, *, seed, draws):
    """Exact squared-coordinate identity; does not subtract DeltaE or prove causes."""
    group_mean = np.empty_like(pred)
    other_image_mean = np.full_like(pred, np.nan)
    patient_other_site_bias_corrected = np.full_like(pred, np.nan)
    for site in np.unique(sites):
        take = np.flatnonzero(sites == site)
        helper.require(np.max(np.abs(target[take] - target[take[0]])) < 5e-5,
                       "Exact within-site target required for MSE decomposition")
        group_mean[take] = pred[take].mean(0)
        if len(take) > 1:
            other_image_mean[take] = (pred[take].sum(0) - pred[take]) / (len(take) - 1)
    for person in np.unique(people):
        own_sites = np.unique(sites[people == person])
        biases = {site: (pred[sites == site] - target[sites == site]).mean(0) for site in own_sites}
        for site in own_sites:
            other = [v for key, v in biases.items() if key != site]
            if other:
                take = sites == site
                patient_other_site_bias_corrected[take] = pred[take] - np.mean(other, 0)
    within_square = (pred - group_mean) ** 2
    bias_square = (group_mean - target) ** 2
    residual_square = (pred - target) ** 2
    cross = 2 * (pred - group_mean) * (group_mean - target)
    helper.require(np.allclose(residual_square.mean(0), within_square.mean(0) + bias_square.mean(0), atol=1e-10, rtol=1e-10),
                   "Within-site decomposition identity failed")
    total = residual_square.mean(0)
    within = within_square.mean(0)
    bias = bias_square.mean(0)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(people == p) for p in np.unique(people)]
    fractions = []
    for _ in range(draws):
        take = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        fractions.append(within_square[take].sum() / residual_square[take].sum())
    oracle = []
    for name, values, definition in [
        ("same_site_mean_prediction_multiframe", group_mean,
         "Mean frozen OOF predictions over all images of a site including this image; privileged grouping and multiple captures, not single-photo deployment."),
        ("same_site_other_images_mean_prediction", other_image_mean,
         "Mean frozen OOF predictions of other same-site images; excludes this image, needs other captures and exact site identity."),
        ("patient_other_unique_sites_residual_bias_correction", patient_other_site_bias_corrected,
         "Subtract mean Lab residual of the patient's other unique sites, each site equally weighted; uses that held-out patient's other instrument targets and is leaky.")]:
        valid = np.isfinite(values).all(1)
        oracle.append({"name": name, "status": DIAGNOSTIC, "definition": definition,
                       "coverage": float(valid.mean()), **helper.metrics(values[valid], target[valid], people[valid])})
    return {"status": DIAGNOSTIC,
            "identity": "image-weighted MSE(pred-target) = MSE(pred-site_mean_pred) + MSE(site_mean_pred-target); target fixed within site",
            "units": "squared Lab coordinate units; not perceptual DeltaE00 variance",
            "mse_Lab": dict(zip("Lab", total.tolist())),
            "within_site_prediction_variation_mse_Lab": dict(zip("Lab", within.tolist())),
            "site_mean_prediction_bias_mse_Lab": dict(zip("Lab", bias.tolist())),
            "within_site_fraction_per_coordinate": dict(zip("Lab", np.divide(within, total, out=np.full(3, np.nan), where=total > 0).tolist())),
            "within_site_fraction_sum_coordinate_mse": float(within.sum() / total.sum()),
            "site_mean_bias_fraction_sum_coordinate_mse": float(bias.sum() / total.sum()),
            "within_site_fraction_patient_bootstrap_95_ci": np.quantile(fractions, [.025, .975]).tolist(),
            "mean_cross_term_Lab": cross.mean(0).tolist(),
            "oracle_evaluations": oracle,
            "interpretation": "An algebraic decomposition of this frozen estimator. Within-site prediction variation combines capture changes and model sensitivity; site mean bias combines estimator bias and shared capture/reference ambiguity. Neither term uniquely identifies a causal data/model share."}, group_mean


def diagnose(args):
    rows = load_json(args.rows)
    lock = load_json(args.diagnostic_lock)
    helper.require(lock.get("experiment") == EXPERIMENT, "Wrong experiment diagnostic lock")
    helper.require(lock.get("frozen_before_oof") is True, "Diagnostic contract must have been frozen before OOF")
    for relative, expected_sha in lock["source_hashes"].items():
        path = ROOT / relative
        helper.require(path.is_file() and helper.sha(path) == expected_sha, f"Locked diagnostic source changed: {relative}")
    for source in (args.capture_report, args.instrument, args.fixed_masks):
        matching = [expected for name, expected in lock["source_hashes"].items() if Path(name).name == source.name]
        helper.require(len(matching) == 1 and helper.sha(source) == matching[0], f"Input differs from locked diagnostic source: {source.name}")
    bootstrap = lock.get("bootstrap", {"seed": 9123, "draws": 2000})
    draws, seed = int(bootstrap["draws"]), int(bootstrap["seed"])
    helper.require(draws >= 100, "Too few cluster bootstrap draws")
    with np.load(args.oof, allow_pickle=False) as raw:
        data = {key: raw[key] for key in raw.files}
    prediction_key = "predicted_lab" if "predicted_lab" in data else "prediction"
    if "predicted_lab" in data and "prediction" in data:
        helper.require(np.array_equal(data["predicted_lab"], data["prediction"]), "Ambiguous prediction keys differ")
    pred, target, patient_alias = helper.align_oof(data, rows, prediction_key, "target", "row_index", "patient")
    helper.require(str(data["experiment"]) == EXPERIMENT, "OOF belongs to another experiment")
    helper.require(str(data["config_sha256"]) == helper.sha(args.config), "OOF frozen configuration hash mismatch")
    helper.require(str(data["mapping_sha256"]) == helper.sha(args.mapping), "OOF frozen mapping hash mismatch")
    mapping = load_json(args.mapping)
    ordering = np.argsort(data["row_index"])
    fold = np.asarray(data["fold"])[ordering]
    helper.require(fold.shape == (966,) and len(np.unique(fold)) == 6, "Expected six complete OOF folds")
    helper.require(np.array_equal(fold, mapping["fold_by_row"]), "OOF fold differs from immutable mapping")
    people = np.array([row["patient"] for row in rows])
    sites = np.array([row["site"] for row in rows])
    for person in np.unique(people):
        helper.require(len(np.unique(fold[people == person])) == 1, "Person split across OOF folds")
    for number in np.unique(fold):
        helper.require(len(np.unique(people[fold == number])) == 4, "Each OOF fold must hold out four people")
    with np.load(args.capture, allow_pickle=False) as raw:
        capture = helper.align_capture(raw, rows, anonymized=args.capture_anonymized)
    error = delta_e00(pred, target)
    helper.require(np.allclose(error, data["delta_e00"][ordering], atol=1e-5, rtol=0), "Saved OOF DeltaE mismatch")
    alias_people = {person: f"P{k+1:02d}" for k, person in enumerate(sorted(set(people)))}
    alias_sites = {site: f"S{k+1:03d}" for k, site in enumerate(sorted(set(sites)))}
    if "site" in data:
        helper.require(np.array_equal(data["site"][ordering], [alias_sites[s] for s in sites]), "OOF site pseudonyms mismatch")
    helper.require(np.array_equal(patient_alias, [alias_people[p] for p in people]), "OOF patient pseudonyms mismatch")
    for field in ("device", "mode", "image_type"):
        if field in data:
            helper.require(np.array_equal(data[field][ordering], [row[field] for row in rows]), f"OOF {field} metadata mismatch")
    report, residuals = helper.analyze(rows, pred, target, capture, bootstrap_draws=draws)
    report["scope"] = "C_REBASE_V1_TRAIN_ONLY_FIXED_RECIPE_OOF_ERROR_FLOOR_AUDIT"
    report["experiment"] = EXPERIMENT
    report["historical_C"] = {"status": "HISTORICAL_NOT_REPRODUCIBLE", "used_for_selection": False,
                              "comparison_is_controlled": False,
                              "reason": "Historical fold assignments and complete training config lost; this is a newly specified experiment."}
    # Persisted exclusions are the OLD non-model audit masks, not selected after OOF.
    with np.load(args.fixed_masks, allow_pickle=False) as masks:
        helper.require(np.array_equal(masks["row_index"], np.arange(966)), "Prior diagnostic mask order mismatch")
        for key in masks.files:
            if key != "row_index":
                helper.require(np.array_equal(masks[key], residuals[f"mask_{key}"]), f"Predefined exclusion changed: {key}")
    expected = lock["prior_counterfactual_thresholds"]
    rules = report["fixed_exclusion_rules"]
    for key, got in [("site_reference_pair_mean_p80", rules["reference_threshold"]),
                     ("site_capture_pair_median_color_mean_p80", rules["capture_threshold"]),
                     ("image_original_high_plus_low_clip_p80", rules["clipping_threshold"])]:
        helper.require(abs(expected[key] - got) < 1e-10, f"Pre-OOF threshold mismatch: {key}")
    report["overall"]["patient_cluster_bootstrap"] = bootstrap_metrics(error, people, seed=seed, draws=draws)
    report["fold_by_fold"] = [{"fold": int(f), **helper.metrics(pred[fold == f], target[fold == f], people[fold == f])} for f in np.unique(fold)]
    capture_report = load_json(args.capture_report)
    cap_sites = {r["site_id"]: r for r in capture_report["per_site"]}
    mean_roi_var = np.array([cap_sites[alias_sites[s]]["original_central_mean_delta_e00"]["mean"]
                            if cap_sites[alias_sites[s]]["pairs"] else np.nan for s in sites], float)
    unique_site_mean_roi_var = np.array([cap_sites[alias_sites[s]]["original_central_mean_delta_e00"]["mean"]
                                      for s in np.unique(sites) if cap_sites[alias_sites[s]]["pairs"]])
    edges = np.asarray(lock["capture_quartiles"]["edges"], float)
    helper.require(lock["capture_quartiles"]["feature"] == "within_site_observed_mean_pair_mean_delta_e00", "Wrong locked capture feature")
    helper.require(np.allclose(edges, np.quantile(unique_site_mean_roi_var, [.25, .5, .75]), atol=1e-10, rtol=0), "Locked quartile thresholds changed")
    q = np.full(len(rows), -1)
    q[np.isfinite(mean_roi_var)] = np.searchsorted(edges, mean_roi_var[np.isfinite(mean_roi_var)], side="left")
    quartiles = []
    for number in range(4):
        take = q == number
        quartiles.append({"quantile": f"Q{number+1}", "coverage": float(take.mean()), "sites": len(set(sites[take])),
                          "boundaries": [None if number == 0 else float(edges[number-1]), None if number == 3 else float(edges[number])],
                          **helper.metrics(pred[take], target[take], people[take]),
                          "global_p95_tail_count": int(np.sum(take & (error >= np.quantile(error, .95)))),
                          "fraction_global_p95_tail": float(np.sum(take & (error >= np.quantile(error, .95))) / np.sum(error >= np.quantile(error, .95)))})
    report["capture_variability_quartiles"] = {"status": DIAGNOSTIC,
        "threshold_unit": "mean within-site pairwise observed central ROI mean Lab D65/2 DeltaE00, unique sites equally weighted",
        "edges": edges.tolist(), "rule": "Q1 <= q25; Q2 (q25,q50]; Q3 (q50,q75]; Q4 > q75; singleton excluded only here",
        "unique_sites_used_to_define_edges": len(unique_site_mean_roi_var), "unknown_images": int(np.sum(q < 0)), "groups": quartiles}
    report["correlations"]["within_site_observed_mean_pair_mean_delta_e00"] = helper.correlation(mean_roi_var, error, people, seed=seed, draws=draws)
    report["patient_centered_correlations"] = {}
    centered_error = error.copy()
    for person in np.unique(people):
        centered_error[people == person] -= error[people == person].mean()
    for name, raw_values in {
        "within_site_observed_mean_pair_mean_delta_e00": mean_roi_var,
        "observed_vs_instrument_euclidean_mixed_observer_proxy": residuals["observed_vs_instrument_euclidean_mixed_observer_proxy"],
        "reference_repeat_pair_mean_delta_e00": residuals["reference_repeat_pair_mean_delta_e00"],
    }.items():
        values = raw_values.copy()
        for person in np.unique(people):
            take = (people == person) & np.isfinite(values)
            values[take] -= values[take].mean()
        association = helper.correlation(values, centered_error, people, seed=seed, draws=draws)
        association["transformation"] = "Subtract each patient's image-weighted mean from feature and DeltaE; descriptive within-person association, not causal adjustment."
        report["patient_centered_correlations"][name] = association
    # Add site-aggregated correlations to show that image replication is not more subjects.
    site_keys = np.unique(sites)
    site_error = np.array([error[sites == site].mean() for site in site_keys])
    site_people = np.array([people[np.flatnonzero(sites == site)[0]] for site in site_keys])
    report["site_equal_weight_correlations"] = {}
    corr_features = {"within_site_observed_mean_pair_mean_delta_e00": mean_roi_var,
                     "within_site_color36_train_z_pair_rms_mean": residuals["within_site_color36_train_z_pair_rms_mean"],
                     "reference_repeat_pair_mean_delta_e00": residuals["reference_repeat_pair_mean_delta_e00"]}
    for key, values in corr_features.items():
        site_values = np.array([values[np.flatnonzero(sites == site)[0]] for site in site_keys])
        result = helper.correlation(site_values, site_error, site_people, seed=seed, draws=draws)
        result["observations"] = result.pop("images")
        result["observation_unit"] = "unique site"
        report["site_equal_weight_correlations"][key] = result
    for name, values in {"device_x_mode": np.array([row["device"]+" | "+row["mode"] for row in rows]),
                         "device_x_image_type": np.array([row["device"]+" | "+row["image_type"] for row in rows])}.items():
        group_stats = []
        for value in np.unique(values):
            take = values == value
            tail = error >= np.quantile(error, .95)
            group_stats.append({"group": str(value), **helper.metrics(pred[take], target[take], people[take]),
                                "tail_count": int(np.sum(tail & take)), "fraction_of_global_tail": float(np.sum(tail & take) / tail.sum()),
                                "tail_rate_within_group": float(np.mean(tail[take]))})
        report["strata"][name] = group_stats
    report["squared_coordinate_error_decomposition"], site_prediction_mean = decomposition(pred, target, sites, people, seed=seed, draws=draws)
    instrument = load_json(args.instrument)
    report["instrument_repeatability"] = instrument["instrument_repeatability"]
    report["site_repeatability"] = instrument["site_repeatability"]
    report["reference_oracles"] = instrument["oracles"]
    reference_noise = np.var(capture["repetitions"], axis=1, ddof=1).mean(0) / 3
    coord_mse = np.mean((pred - target)**2, axis=0)
    report["iid_reference_mean_noise_proxy"] = {
        "status": DIAGNOSTIC, "assumptions_verified": False,
        "assumptions": "Three independent, zero-mean, identically distributed readings of the same latent color; not established by released metadata.",
        "image_weighted_mean_variance_of_triplet_mean_Lab": dict(zip("Lab", reference_noise.tolist())),
        "ratio_to_observed_coordinate_mse_Lab": dict(zip("Lab", (reference_noise / coord_mse).tolist())),
        "ratio_to_sum_coordinate_mse": float(reference_noise.sum() / coord_mse.sum()),
        "interpretation": "Conditional scale comparison only, not a causal noise share, latent error floor, or valid quantity to subtract from DeltaE00."}
    report["capture_reference_audit_summaries"] = {
        "all_within_site_pairs": capture_report["all_pairs"],
        "counts": capture_report["counts"],
        "instrument_source": str(args.instrument.name), "capture_source": str(args.capture_report.name)}
    for rec in report["top50"]:
        index = int(rec["row_id"][1:]) - 1
        rec["image_id"] = f"I{index+1:04d}"
        rec["fold"] = int(fold[index])
        rec["within_site_observed_mean_pair_mean_delta_e00"] = float(mean_roi_var[index]) if np.isfinite(mean_roi_var[index]) else None
        rec["capture_variability_quartile"] = f"Q{q[index]+1}" if q[index] >= 0 else "unknown_singleton"
        for column in range(36):
            rec[f"color36_{column:02d}"] = float(capture["color36"][index, column])
    report["limitations"] = [line.replace("Frozen C", "Frozen C_REBASE_V1").replace("No model retraining or model selection.", "This analyzer does not fit models; the separate C_REBASE_V1 recipe was frozen before training.")
                             for line in report["limitations"]]
    report["limitations"].extend([
        "No validated causal allocation among instrument, capture and estimator errors is identifiable from this observational dataset.",
        "Same-site observations cross acquisition modes; no repeated independent reference sessions or same-mode photograph replicates were established.",
        "Capture quartiles and reference exclusions are privileged diagnostics, not deployable photo-quality gates.",
        "Clinical close-up/dermoscopy TRAIN OOF is not an independent selfie/instrument-reference test."])
    report["provenance"] = {
        "oof_sha256": helper.sha(args.oof), "rows_sha256": helper.sha(args.rows),
        "capture_sha256": helper.sha(args.capture), "capture_report_sha256": helper.sha(args.capture_report),
        "instrument_sha256": helper.sha(args.instrument), "fixed_masks_sha256": helper.sha(args.fixed_masks),
        "diagnostic_lock_sha256": helper.sha(args.diagnostic_lock), "script_sha256": helper.sha(__file__),
        "config_sha256": helper.sha(args.config), "mapping_sha256": helper.sha(args.mapping),
        "locked_diagnostic_source_hashes": lock["source_hashes"], "locked_sources_verified": True,
        "helper_script_sha256": helper.sha(ROOT / "scripts/error_floor/c_oof_audit.py"),
        "original_source_hashes": instrument.get("provenance", {}),
        "alignment": "Exact original row permutation, target equality, pseudonym/site metadata checks, one fold per patient and four patients per fold",
        "reserved_endpoints_loaded": False, "models_trained_by_this_script": 0,
        "bootstrap": bootstrap, "oof_prediction_key": prediction_key}
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "c_rebase_v1_error_audit.json", report)
    with (args.output / "top50_anonymized.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["top50"][0]))
        writer.writeheader(); writer.writerows(report["top50"])
    all_arrays = {key: value for key, value in residuals.items() if key not in ("image", "patient", "site")}
    all_arrays.update(row_index=np.arange(len(rows)), image=np.array([f"I{i+1:04d}" for i in range(len(rows))]),
                      patient=np.array([alias_people[p] for p in people]), site=np.array([alias_sites[s] for s in sites]),
                      fold=fold, color36=capture["color36"], within_site_observed_mean_pair_mean_delta_e00=mean_roi_var,
                      capture_variability_quartile=q, site_mean_prediction=site_prediction_mean)
    np.savez_compressed(args.output / "c_rebase_v1_diagnostic_rows_anonymized.npz", **all_arrays)
    # Flat, reviewable tables accompany the complete machine-readable report.
    stratum_rows = []
    for dimension, groups in report["strata"].items():
        for group in groups:
            stratum_rows.append({"dimension": dimension, "group": group["group"], "people": group["people"],
                                  **group["delta_e00"], "tail_count": group["tail_count"],
                                  "fraction_of_global_tail": group["fraction_of_global_tail"],
                                  "person_balanced_mean": group["person_balanced_mean"]})
    with (args.output / "strata_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stratum_rows[0])); writer.writeheader(); writer.writerows(stratum_rows)
    correlation_rows = []
    for definition in ("correlations", "patient_centered_correlations", "site_equal_weight_correlations"):
        for feature, values in report[definition].items():
            correlation_rows.append({"definition": definition, "feature": feature, "observations": values.get("images", values.get("observations")),
                                     "people": values["people"], "pearson": values["pearson"], "spearman": values["spearman"],
                                     "pearson_ci_low": values.get("pearson_95_ci", [None,None])[0],
                                     "pearson_ci_high": values.get("pearson_95_ci", [None,None])[1],
                                     "spearman_ci_low": values.get("spearman_95_ci", [None,None])[0],
                                     "spearman_ci_high": values.get("spearman_95_ci", [None,None])[1]})
    with (args.output / "correlations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(correlation_rows[0])); writer.writeheader(); writer.writerows(correlation_rows)
    save_json(args.output / "analysis_output_sha256.json", {
        name: helper.sha(args.output / name) for name in ("c_rebase_v1_error_audit.json", "top50_anonymized.csv", "c_rebase_v1_diagnostic_rows_anonymized.npz", "strata_summary.csv", "correlations.csv")})
    print(json.dumps(clean({"experiment": EXPERIMENT, "overall": report["overall"],
                           "quartiles": quartiles, "decomposition": report["squared_coordinate_error_decomposition"]}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("oof", "rows", "capture", "capture-report", "instrument", "fixed-masks", "diagnostic-lock", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/c_rebase_v1/config.lock.json")
    parser.add_argument("--mapping", type=Path, default=ROOT / "experiments/c_rebase_v1/person_folds.lock.json")
    parser.add_argument("--capture-anonymized", action="store_true")
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
