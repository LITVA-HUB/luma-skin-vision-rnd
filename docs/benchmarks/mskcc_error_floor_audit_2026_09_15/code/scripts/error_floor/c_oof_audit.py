"""Offline audit of saved C OOF predictions. No fitting or downloads.

Inspect NPZ keys first; run requires explicit prediction/target/index/group keys.
An integer index is the ZERO-BASED position in original image-ID-sorted TRAIN.
This tool cannot authenticate how a supplied prediction was trained: archive
provenance remains necessary. It does validate row, target and group alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from luma_skin_vision.color import delta_e00


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype)} for k in data.files}


def align_oof(data, rows, prediction_key, target_key, index_key, group_key, *, expected_n=966):
    require(rows == sorted(rows, key=lambda r: r["image"]), "Mapping must be original image-ID-sorted TRAIN")
    require(len(rows) == expected_n and len({r["image"] for r in rows}) == expected_n,
            "TRAIN mapping count or image uniqueness mismatch")
    for name in (prediction_key, target_key, index_key, group_key):
        require(name in data, f"Explicit NPZ key missing: {name}")
    index = np.asarray(data[index_key])
    require(index.shape == (expected_n,) and index.dtype.kind in "iu", "Index key must contain 0-based integer row indices")
    require(np.array_equal(np.sort(index), np.arange(expected_n)), "Indices must be a complete unique permutation of TRAIN rows")
    pred, target = np.asarray(data[prediction_key], dtype=float), np.asarray(data[target_key], dtype=float)
    groups = np.asarray(data[group_key])
    require(pred.shape == target.shape == (expected_n, 3), "Predictions and targets must have shape (N,3)")
    require(groups.shape == (expected_n,) and groups.dtype.kind in "USiu", "Group key must be scalar string/integer labels per image")
    require(np.isfinite(pred).all() and np.isfinite(target).all(), "Nonfinite predictions/targets")
    order = np.argsort(index)
    pred, target, groups = pred[order], target[order], groups[order].astype(str)
    original_target = np.asarray([r["target"] for r in rows], dtype=float)
    require(np.allclose(target, original_target, atol=5e-5, rtol=0), "OOF target differs from original TRAIN target at asserted index")
    people = np.array([r["patient"] for r in rows])
    for group in np.unique(groups):
        require(len(set(people[groups == group])) == 1, "An OOF group mixes original patients")
    for patient in np.unique(people):
        require(len(set(groups[people == patient])) == 1, "Original patient split across OOF group labels")
    return pred, original_target, groups


def align_capture(data, rows, *, anonymized=False):
    names = ("image", "patient", "site", "device", "mode", "image_type", "anatomic_site")
    required = names + ("target", "repetitions", "original_median_lab", "original_mean_lab", "original_median_rgb",
                        "color36", "diagnostic", "diagnostic_names", "width", "height")
    for k in required:
        require(k in data, f"Capture NPZ key missing: {k}")
    ids = np.asarray(data["image"]).astype(str)
    require(ids.shape == (len(rows),) and len(set(ids)) == len(rows), "Capture snapshot incomplete or duplicated")
    if anonymized:
        require("row_index" in data, "Anonymized capture requires original row_index")
        index = np.asarray(data["row_index"])
        require(index.dtype.kind in "iu" and np.array_equal(np.sort(index), np.arange(len(rows))),
                "Anonymized capture index must be complete original-index permutation")
        order = np.argsort(index)
    else:
        require(set(ids) == {r["image"] for r in rows}, "Capture images differ from original TRAIN mapping")
        lookup = {x: i for i, x in enumerate(ids)}
        order = np.array([lookup[r["image"]] for r in rows])
    out = {k: np.asarray(data[k])[order] for k in required if k != "diagnostic_names"}
    out["diagnostic_names"] = np.asarray(data["diagnostic_names"])
    patient_alias = {p: f"P{i+1:02d}" for i, p in enumerate(sorted({r["patient"] for r in rows}))}
    site_alias = {s: f"S{i+1:03d}" for i, s in enumerate(sorted({r["site"] for r in rows}))}
    for k in names:
        expected = np.array([r[k] for r in rows])
        if anonymized and k == "image": expected = np.array([f"I{i+1:04d}" for i in range(len(rows))])
        if anonymized and k == "patient": expected = np.array([patient_alias[r["patient"]] for r in rows])
        if anonymized and k == "site": expected = np.array([site_alias[r["site"]] for r in rows])
        require(np.array_equal(out[k].astype(str), expected), f"Capture {k} mismatch")
    require(np.allclose(out["target"], [r["target"] for r in rows], atol=5e-5, rtol=0), "Capture target mismatch")
    require(np.array_equal(out["repetitions"], np.array([r["repetitions"] for r in rows])), "Capture repetitions mismatch")
    for k in required:
        if k in names or k == "diagnostic_names":
            continue
        require(np.isfinite(out[k]).all(), f"Nonfinite capture feature: {k}")
    require(out["color36"].shape == (len(rows), 36), "Color36 shape mismatch")
    require(out["diagnostic"].shape == (len(rows), len(out["diagnostic_names"])), "Diagnostic schema mismatch")
    return out


def summary(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None,
                "max": None, "fraction_gt_5": None, "fraction_gt_10": None}
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.quantile(x, .9)), "p95": float(np.quantile(x, .95)), "max": float(x.max()),
            "fraction_gt_5": float(np.mean(x > 5)), "fraction_gt_10": float(np.mean(x > 10))}


def metrics(pred, target, people):
    error = delta_e00(pred, target)
    if not len(error):
        return {"delta_e00": summary(error), "people": 0}
    residual = pred - target
    pmeans = [float(error[people == p].mean()) for p in np.unique(people)]
    pmedians = [float(np.median(error[people == p])) for p in np.unique(people)]
    weights = np.array([1 / np.sum(people == p) for p in people])
    order = np.argsort(error)
    equal_person_median = error[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]]
    return {"delta_e00": summary(error), "people": len(pmeans),
            "mae_lab": dict(zip("Lab", np.mean(np.abs(residual), axis=0).tolist())),
            "rmse_lab": dict(zip("Lab", np.sqrt(np.mean(residual ** 2, axis=0)).tolist())),
            "person_balanced_mean": float(np.mean(pmeans)),
            "mean_of_person_medians": float(np.mean(pmedians)),
            "equal_person_weighted_median": float(equal_person_median)}


def correlation(x, error, people, seed=9123, draws=1000):
    finite = np.isfinite(x) & np.isfinite(error)
    x, y, people = np.asarray(x)[finite], np.asarray(error)[finite], np.asarray(people)[finite]
    result = {"images": len(x), "people": len(set(people)), "bootstrap_unit": "patient", "bootstrap_draws": draws}
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result | {"pearson": None, "spearman": None, "reason": "constant_or_insufficient"}
    result.update(pearson=float(np.corrcoef(x, y)[0, 1]),
                  spearman=float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]))
    unique = sorted(set(people))
    groups = [np.flatnonzero(people == p) for p in unique]
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if np.ptp(x[i]) and np.ptp(y[i]):
            boot.append([np.corrcoef(x[i], y[i])[0, 1], np.corrcoef(rankdata(x[i]), rankdata(y[i]))[0, 1]])
    if boot:
        interval = np.quantile(boot, [.025, .975], axis=0)
        result["pearson_95_ci"] = interval[:, 0].tolist()
        result["spearman_95_ci"] = interval[:, 1].tolist()
    result["valid_bootstrap_draws"] = len(boot)
    result["interpretation"] = "Exploratory association, patient-cluster intervals; no multiplicity correction or causal attribution"
    return result


def variability(rows, capture):
    site_indices = defaultdict(list)
    for i, row in enumerate(rows):
        site_indices[(row["patient"], row["site"])].append(i)
    n = len(rows)
    ref = np.full(n, np.nan)
    image = np.full(n, np.nan)
    color = np.full(n, np.nan)
    reference_unique, capture_unique = [], []
    scale = capture["color36"].std(0)
    scale[scale < 1e-12] = 1.
    for indices in site_indices.values():
        r = capture["repetitions"][indices[0]]
        rmean = float(np.mean([delta_e00(r[i], r[j]) for i, j in itertools.combinations(range(3), 2)]))
        ref[indices] = rmean
        reference_unique.append(rmean)
        if len(indices) < 2:
            continue
        image_mean = float(np.mean([delta_e00(capture["original_median_lab"][i], capture["original_median_lab"][j])
                                   for i, j in itertools.combinations(indices, 2)]))
        color_mean = float(np.mean([np.sqrt(np.mean(((capture["color36"][i] - capture["color36"][j]) / scale) ** 2))
                                   for i, j in itertools.combinations(indices, 2)]))
        image[indices], color[indices] = image_mean, color_mean
        capture_unique.append(image_mean)
    return {"reference_repeat_pair_mean_delta_e00": ref,
            "within_site_observed_median_pair_mean_delta_e00": image,
            "within_site_color36_train_z_pair_rms_mean": color}, reference_unique, capture_unique


def analyze(rows, pred, target, capture, *, bootstrap_draws=1000):
    n = len(rows)
    people = np.array([r["patient"] for r in rows])
    p_alias = {p: f"P{i+1:02d}" for i, p in enumerate(sorted(set(people)))}
    s_alias = {s: f"S{i+1:03d}" for i, s in enumerate(sorted({r["site"] for r in rows}))}
    aliases = np.array([f"R{i+1:04d}" for i in range(n)])
    error = delta_e00(pred, target)
    global_tail = float(np.quantile(error, .95))
    tail = error >= global_tail
    diag = {str(k): capture["diagnostic"][:, j] for j, k in enumerate(capture["diagnostic_names"])}
    var, reference_unique, capture_unique = variability(rows, capture)
    # Explicitly a coordinate discrepancy proxy across 2-degree/10-degree conventions.
    discrepancy = capture["original_median_lab"] - target
    features = {"target_L": target[:, 0], "observed_median_L_D65_2deg": capture["original_median_lab"][:, 0],
                **diag, **var,
                "observed_minus_instrument_L_mixed_observer_proxy": discrepancy[:, 0],
                "observed_minus_instrument_a_mixed_observer_proxy": discrepancy[:, 1],
                "observed_minus_instrument_b_mixed_observer_proxy": discrepancy[:, 2],
                "observed_vs_instrument_euclidean_mixed_observer_proxy": np.linalg.norm(discrepancy, axis=1)}
    strata = {}
    fields = {k: np.array([r[k] for r in rows]) for k in
              ("patient", "site", "device", "mode", "image_type", "anatomic_site")}
    fields["target_L_bins"] = np.array([f"[{int(v//10)*10},{int(v//10)*10+10})" for v in target[:, 0]])
    fields["observed_brightness_bins"] = np.array([f"[{min(int(v*5),4)*.2:.1f},{(min(int(v*5),4)+1)*.2:.1f}]"
                                                for v in diag["brightness_mean_encoded_RGB"]])
    high = diag["original_high_clip_pixel_fraction"]
    low = diag["original_low_clip_pixel_fraction"]
    clip_proxy = high + low
    # Sum can count a pixel twice; explicitly a proxy, not claimed union coverage.
    fields["original_clip_proxy_bins"] = np.select([clip_proxy == 0, clip_proxy <= .01, clip_proxy <= .05],
                                                  ["0", "(0,.01]", "(.01,.05]"], default=">.05")
    for field, values in fields.items():
        groups = []
        for value in sorted(set(values)):
            take = values == value
            label = p_alias[value] if field == "patient" else s_alias[value] if field == "site" else str(value)
            groups.append({"group": label, **metrics(pred[take], target[take], people[take]),
                           "global_p95_threshold": global_tail, "tail_count": int(np.sum(tail & take)),
                           "tail_rate_within_group": float(np.mean(tail[take])),
                           "fraction_of_global_tail": float(np.sum(tail & take) / tail.sum())})
        strata[field] = groups
    correlations = {name: correlation(value, error, people, draws=bootstrap_draws)
                    for name, value in features.items()}
    # Thresholds determined from non-C variability only, frozen in source before the
    # missing OOF archive arrives. Ties retained; actual coverage is not forced to80%.
    ref_threshold = float(np.quantile(reference_unique, .8))
    capture_threshold = float(np.quantile(capture_unique, .8)) if capture_unique else None
    clip_threshold = float(np.quantile(clip_proxy, .8))
    masks = {
        "all_TRAIN": np.ones(n, dtype=bool),
        "lower_80pct_site_reference_variability": var["reference_repeat_pair_mean_delta_e00"] <= ref_threshold,
        "lower_80pct_site_capture_variability": np.isfinite(var["within_site_observed_median_pair_mean_delta_e00"]) &
                                      (var["within_site_observed_median_pair_mean_delta_e00"] <= (capture_threshold if capture_threshold is not None else -1)),
        "lower_80pct_image_clipping_proxy": clip_proxy <= clip_threshold}
    masks["reference_and_capture_stable"] = masks["lower_80pct_site_reference_variability"] & masks["lower_80pct_site_capture_variability"]
    masks["reference_capture_and_clipping_stable"] = masks["reference_and_capture_stable"] & masks["lower_80pct_image_clipping_proxy"]
    counterfactual = []
    for name, take in masks.items():
        counterfactual.append({"name": name, "status": "LEAKY_DIAGNOSTIC_ONLY", "coverage": float(take.mean()),
                               "accepted_images": int(take.sum()), **metrics(pred[take], target[take], people[take]),
                               "interpretation": "Descriptive exclusion, original full-data metric retained; not an achievable causal error floor"})
    references = []
    reps = capture["repetitions"]
    for k in range(3):
        changed = (reps.sum(1) - reps[:, k]) / 2
        references.append({"name": f"target_mean_omitting_recorded_assessment_{k+1}",
                           "status": "LEAKY_DIAGNOSTIC_ONLY", **metrics(pred, changed, people),
                           "target_perturbation_delta_e00": summary(delta_e00(target, changed)),
                           "interpretation": "Frozen C predictions versus alternate same-site reference aggregation; latent true color unknown"})
    changed = np.median(reps, axis=1)
    references.append({"name": "coordinate_median_of_three_recorded_assessments", "status": "LEAKY_DIAGNOSTIC_ONLY",
                       **metrics(pred, changed, people), "target_perturbation_delta_e00": summary(delta_e00(target, changed)),
                       "interpretation": "Alternative reference, not removing known instrument noise"})
    order = np.lexsort((np.arange(n), -error))[:50]
    records = []
    for i in order:
        row = rows[i]
        rec = {"row_id": aliases[i], "person_id": p_alias[row["patient"]], "site_id": s_alias[row["site"]],
               "delta_e00": float(error[i]), "device": row["device"], "mode": row["mode"],
               "image_type": row["image_type"], "anatomic_site": row["anatomic_site"],
               "width": int(capture["width"][i]), "height": int(capture["height"][i])}
        for k, c in enumerate("Lab"):
            rec.update({f"target_{c}": float(target[i, k]), f"predicted_{c}": float(pred[i, k]),
                        f"residual_{c}": float(pred[i, k] - target[i, k]),
                        f"observed_median_{c}_D65_2deg": float(capture["original_median_lab"][i, k])})
        for k, c in enumerate("RGB"):
            rec[f"observed_median_{c}_encoded_sRGB"] = float(capture["original_median_rgb"][i, k])
        rec.update({name: float(value[i]) if np.isfinite(value[i]) else None for name, value in features.items()})
        records.append(rec)
    report = {"scope": "FROZEN_C_OOF_TRAIN_ONLY_DESCRIPTIVE_ERROR_AUDIT",
              "counts": {"images": n, "people": len(set(people)), "sites": len(s_alias)},
              "overall": metrics(pred, target, people), "global_p95_tail": {"threshold": global_tail, "images": int(tail.sum())},
              "strata": strata, "correlations": correlations, "top50": records,
              "counterfactual_exclusions": counterfactual, "reference_perturbations": references,
              "fixed_exclusion_rules": {"reference_threshold": ref_threshold, "reference_units": "mean of3within-site instrument pair DE00",
                                        "capture_threshold": capture_threshold, "capture_units": "mean original central-median observed DE00 over same-site photo pairs",
                                        "clipping_threshold": clip_threshold, "clipping_units": "original high-pixel fraction + low-pixel fraction, potential double counting",
                                        "quantile": .8, "ties": "included", "single_image_sites_capture_rule": "ineligible, not presumed stable",
                                        "selection_used_C_error": False},
              "limitations": ["No model retraining or model selection.",
                              "Patient/device/anatomical associations are confounded; 24 people provide limited precision.",
                              "Source image Lab D65/2 versus instrument D65/10 differences are coordinate proxies, never physical DeltaE00.",
                              "Within-site variability needs other photos; these diagnostic variables are unavailable to single-photo deployment.",
                              "Reference exclusions use measured targets and are leaky diagnostics.",
                              "No subtraction of instrument DeltaE00; no causal fraction or identifiable attainable floor asserted.",
                              "The supplied OOF training provenance must be checked separately from these alignment checks.",
                              "Swapping a falsely declared index between photos with identical site targets cannot be detected from targets alone; the original OOF index contract and archive provenance are required."]}
    return report, {"image": np.array([r["image"] for r in rows]), "patient": people,
                    "site": np.array([r["site"] for r in rows]), "prediction": pred, "target": target,
                    "residual_lab": pred - target, "delta_e00": error, **features,
                    **{f"mask_{k}": v for k, v in masks.items()}}


def run(args):
    rows = json.loads(args.rows.read_bytes())
    require(len({r["patient"] for r in rows}) == 24 and len({r["site"] for r in rows}) == 248,
            "Original TRAIN patient/site counts differ")
    with np.load(args.oof, allow_pickle=False) as data:
        pred, target, _ = align_oof(data, rows, args.prediction_key, args.target_key, args.index_key, args.group_key)
    with np.load(args.capture, allow_pickle=False) as data:
        capture = align_capture(data, rows, anonymized=args.capture_anonymized)
    report, residuals = analyze(rows, pred, target, capture, bootstrap_draws=args.bootstrap_draws)
    with np.load(args.fixed_masks, allow_pickle=False) as masks:
        require(np.array_equal(masks["row_index"], np.arange(len(rows))), "Frozen diagnostic mask row order differs")
        expected_names = {k.removeprefix("mask_") for k in residuals if k.startswith("mask_")}
        require(set(masks.files) - {"row_index"} == expected_names, "Frozen diagnostic mask definitions differ")
        for name in expected_names:
            require(np.array_equal(masks[name], residuals[f"mask_{name}"]), f"Frozen diagnostic mask changed: {name}")
    report["provenance"] = {"oof_sha256": sha(args.oof), "rows_sha256": sha(args.rows), "capture_sha256": sha(args.capture),
                            "fixed_masks_sha256": sha(args.fixed_masks), "fixed_masks_exactly_verified": True,
                            "capture_anonymized": args.capture_anonymized,
                            "script_sha256": sha(__file__), "explicit_keys": {"prediction": args.prediction_key,
                            "target": args.target_key, "index": args.index_key, "group": args.group_key},
                            "alignment": "Complete original index permutation + numeric target equality + patient/group bijection + capture ID metadata join",
                            "target_atol": 5e-5, "models_trained": 0, "reserved_endpoints_loaded": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "c_error_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "top50_anonymized.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["top50"][0]))
        writer.writeheader()
        writer.writerows(report["top50"])
    np.savez_compressed(args.output / "c_residuals.private.npz", **residuals)
    print(json.dumps({"scope": report["scope"], "counts": report["counts"], "overall": report["overall"]}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="command", required=True)
    view = modes.add_parser("inspect")
    view.add_argument("npz", type=Path)
    execute = modes.add_parser("run")
    for name in ("oof", "rows", "capture", "fixed-masks", "output"):
        execute.add_argument(f"--{name}", type=Path, required=True)
    execute.add_argument("--capture-anonymized", action="store_true")
    for name in ("prediction-key", "target-key", "index-key", "group-key"):
        execute.add_argument(f"--{name}", required=True)
    execute.add_argument("--bootstrap-draws", type=int, default=1000)
    args = p.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect(args.npz), indent=2))
    else:
        require(args.bootstrap_draws >= 100, "Use at least100cluster-bootstrap draws")
        run(args)


if __name__ == "__main__":
    main()
