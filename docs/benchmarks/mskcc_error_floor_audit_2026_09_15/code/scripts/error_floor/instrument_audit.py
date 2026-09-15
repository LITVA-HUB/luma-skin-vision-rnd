"""TRAIN-only original-reference audit; never loads model or held-out endpoints.

Run: PYTHONPATH=src python scripts/error_floor/instrument_audit.py --output DIR
This script does not call skin_mskcc_data.prepare() or rewrite old receipts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from luma_skin_vision.color import delta_e00


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summary(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return {"n": 0, "mean": None, "median": None, "p90": None,
                "p95": None, "max": None}
    assert np.isfinite(x).all()
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.quantile(x, .9)), "p95": float(np.quantile(x, .95)),
            "max": float(x.max()), "fraction_gt_2": float(np.mean(x > 2)),
            "fraction_gt_5": float(np.mean(x > 5)), "fraction_gt_10": float(np.mean(x > 10))}


def repetitions(row):
    # Call only after an exact TRAIN image/site membership check.
    value = np.array([[float(row[f"{c}_{k}"]) for c in "lab"] for k in (1, 2, 3)])
    assert np.isfinite(value).all()
    assert np.allclose(value.mean(0), [float(row[f"average_{c}"]) for c in "lab"],
                       atol=.051, rtol=0)
    return value


def mean_ci_by_people(values, groups, seed=73019, draws=10000):
    # Equal-weight patients, each patient's statistic averages its unique sites/pairs.
    means = np.array([np.mean(np.asarray(values)[np.asarray(groups) == p])
                      for p in sorted(set(groups))])
    rng = np.random.default_rng(seed)
    boots = means[rng.integers(0, len(means), (draws, len(means)))].mean(1)
    return {"patient_balanced_mean": float(means.mean()),
            "patient_bootstrap_mean_95_ci": np.quantile(boots, [.025, .975]).tolist(),
            "bootstrap_unit": "patient", "draws": draws, "seed": seed}


def evaluate_oracle(name, targets, predictions, patients, definition):
    eligible = np.isfinite(predictions).all(1)
    errors = delta_e00(targets[eligible], predictions[eligible])
    return {"name": name, "status": "LEAKY_DIAGNOSTIC_ONLY", "definition": definition,
            "eligible_images": int(eligible.sum()), "total_images": len(targets),
            "coverage": float(eligible.mean()), "eligible_people": len(set(patients[eligible])),
            "delta_e00": summary(errors),
            "patient_balanced": mean_ci_by_people(errors, patients[eligible]) if eligible.any() else None}


def main(output):
    raw = ROOT / "data/public/mskcc_skin_v1"
    prov = ROOT / "docs/data/provenance/mskcc_skin_v1"
    acquisition = json.loads((prov / "acquisition.json").read_bytes())
    source_hashes = {}
    for rec in acquisition["files"]:
        actual = digest(raw / rec["file"])
        if actual != rec["sha256"]:
            raise ValueError(f"Original acquisition SHA mismatch: {rec['file']}")
        source_hashes[rec["file"]] = actual
    split = json.loads((prov / "split_receipt.json").read_bytes())
    assert all(source_hashes[k] == v for k, v in split["source_sha256"].items())

    spec = importlib.util.spec_from_file_location("original_mskcc_data", ROOT / "scripts/skin_mskcc_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only metadata fields are retained for role assignment. No numeric endpoints
    # or sensitive personal demographic fields are decoded outside TRAIN.
    people = {r["patient_id"]: r["dermatoscope"] for r in read(raw / "s1.csv")}
    sites = {r["tag_id"]: {k: r[k] for k in ("patient_id", "lesion_id", "type", "anatomic_site")}
             for r in read(raw / "s2.csv")}
    meta = {r["isic_id"]: {k: r[k] for k in ("patient_id", "lesion_id", "image_type", "copyright_license")}
            for r in read(raw / "mskcc-skin-tone-labeling-dataset.csv")}
    s7 = read(raw / "s7.csv")
    cohort_devices = {sites[r["tag_id"]]["patient_id"]: people[sites[r["tag_id"]]["patient_id"]]
                      for r in s7 if r["isic_id"] != "not-available"}
    roles = module.patient_roles(cohort_devices)
    assert sum(v == "train" for v in roles.values()) == 24
    rows, s7_train = [], {}
    for row in s7:
        if row["isic_id"] == "not-available":
            continue
        site = sites[row["tag_id"]]
        patient = site["patient_id"]
        if roles[patient] != "train":
            continue
        m = meta[row["isic_id"]]
        assert m["patient_id"] == patient and m["lesion_id"] == site["lesion_id"]
        assert m["copyright_license"] == "CC-BY" and row["type"] == "normal-skin"
        reps = repetitions(row)
        rows.append({"image": row["isic_id"], "patient": patient, "site": row["tag_id"],
                     "device": people[patient], "mode": row["dermoscopic_type"],
                     "image_type": m["image_type"], "anatomic_site": row["anatomic_site"],
                     "target": reps.mean(0).tolist(), "repetitions": reps.tolist(),
                     "published_image_lab": [float(row[f"img_{c}"]) for c in "lab"]})
        s7_train[row["isic_id"]] = row
    rows.sort(key=lambda r: r["image"])
    assert (len(rows), len({r["patient"] for r in rows}), len({r["site"] for r in rows})) == (966, 24, 248)
    assert len({r["image"] for r in rows}) == 966
    train_sites = {r["site"] for r in rows}
    by_site = defaultdict(list)
    for row in rows:
        by_site[row["site"]].append(row)
    source_s4 = defaultdict(list)
    for row in read(raw / "s4.csv"):
        if row["tag_id"] in train_sites:
            source_s4[row["tag_id"]].append(row)
    assert set(source_s4) == train_sites
    s4_reps = {}
    for tag, observations in source_s4.items():
        # Distinct repeat-triplet rows per site cannot be silently collapsed.
        assert len(observations) == 1, "Multiple reference acquisitions require an explicit session key"
        s4_reps[tag] = repetitions(observations[0])
        for image_row in by_site[tag]:
            assert np.array_equal(s4_reps[tag], image_row["repetitions"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_train_rows.private.json").write_text(json.dumps(rows, indent=2) + "\n")

    ordered_sites = sorted(train_sites)
    ordered_people = sorted({r["patient"] for r in rows})
    p_alias = {p: f"P{i+1:02d}" for i, p in enumerate(ordered_people)}
    site_alias = {s: f"S{i+1:03d}" for i, s in enumerate(ordered_sites)}
    site_reps = np.array([s4_reps[s] for s in ordered_sites])
    site_targets = site_reps.mean(1)
    site_patients = np.array([sites[s]["patient_id"] for s in ordered_sites])
    pairs = list(itertools.combinations(range(3), 2))
    pair_diffs = np.stack([site_reps[:, j] - site_reps[:, i] for i, j in pairs], 1)
    pair_de = np.stack([delta_e00(site_reps[:, i], site_reps[:, j]) for i, j in pairs], 1)
    repeat_sd = site_reps.std(1, ddof=1)
    single_vs_mean = delta_e00(site_reps, site_targets[:, None])
    loo_mean = (site_reps.sum(1)[:, None] - site_reps) / 2
    loo_errors = delta_e00(site_reps, loo_mean)
    target_sensitivity = delta_e00(loo_mean, site_targets[:, None])

    stratum_tables = {}
    for field in ("device", "anatomic_site"):
        labels = np.array([by_site[s][0][field] for s in ordered_sites])
        stratum_tables[field] = [{"group": value, "sites": int(np.sum(labels == value)),
                                 "people": len(set(site_patients[labels == value])),
                                 "pairwise_delta_e00": summary(pair_de[labels == value])}
                                for value in sorted(set(labels))]

    same_site_image_de = []
    image_count_by_site = Counter(len(v) for v in by_site.values())
    for group in by_site.values():
        for left, right in itertools.combinations(group, 2):
            same_site_image_de.append(float(delta_e00(left["target"], right["target"])))
    targets = np.array([r["target"] for r in rows])
    patients = np.array([r["patient"] for r in rows])
    target_sites = np.array([r["site"] for r in rows])
    devices = np.array([r["device"] for r in rows])
    published_lab = np.array([r["published_image_lab"] for r in rows])
    loc = np.array([by_site[s][0]["anatomic_site"] for s in ordered_sites])
    dev = np.array([by_site[s][0]["device"] for s in ordered_sites])
    result_predictions = {k: np.full(targets.shape, np.nan) for k in
                          ("same_site_leave_image_out", "same_site_nearest_observed",
                           "patient_other_sites", "patient_same_camera_other_sites",
                           "same_camera_anatomic_region_other_sites")}
    for i, row in enumerate(rows):
        other = np.flatnonzero((target_sites == row["site"]) & (np.arange(len(rows)) != i))
        if len(other):
            result_predictions["same_site_leave_image_out"][i] = targets[other].mean(0)
            # Selection uses authors' image median Lab Euclidean coordinates, not target.
            near = other[np.argmin(np.linalg.norm(published_lab[other] - published_lab[i], axis=1))]
            result_predictions["same_site_nearest_observed"][i] = targets[near]
        for name, eligible in [
            ("patient_other_sites", (site_patients == row["patient"]) & (np.array(ordered_sites) != row["site"])),
            ("patient_same_camera_other_sites", (site_patients == row["patient"]) & (dev == row["device"]) & (np.array(ordered_sites) != row["site"])),
            ("same_camera_anatomic_region_other_sites", (dev == row["device"]) & (loc == row["anatomic_site"]) & (np.array(ordered_sites) != row["site"]))]:
            if eligible.any():
                result_predictions[name][i] = site_targets[eligible].mean(0)
    definitions = {
        "same_site_leave_image_out": "Other image targets at exact patient+tag_id; shares the same S4 reference triplet, so zero is tautological.",
        "same_site_nearest_observed": "Other exact-site image nearest in authors' published median image Lab Euclidean coordinates; reference triplet is shared.",
        "patient_other_sites": "Equal-site mean of other unique measurement sites of same person, excluding all images sharing target site's reference.",
        "patient_same_camera_other_sites": "Same-person/same-device other unique sites; camera is person-confounded, so equals patient oracle.",
        "same_camera_anatomic_region_other_sites": "Mean of other unique TRAIN sites with same device and recorded anatomical region; target's exact tag_id excluded; references from other people are used."}
    oracles = [evaluate_oracle(name, targets, pred, patients, definitions[name])
               for name, pred in result_predictions.items()]
    oracles.append({"name": "same_site_leave_reference_observation_out", "status": "LEAKY_DIAGNOSTIC_ONLY",
                    "eligible_images": 0, "total_images": 966, "coverage": 0,
                    "delta_e00": summary([]),
                    "reason": "Each exact site has one S4 triplet only; excluding its shared reference removes all same-site neighbours."})
    per_site = []
    for j, tag in enumerate(ordered_sites):
        per_site.append({"site_id": site_alias[tag], "person_id": p_alias[sites[tag]["patient_id"]],
                         "images": len(by_site[tag]), "device": by_site[tag][0]["device"],
                         "anatomic_site": by_site[tag][0]["anatomic_site"],
                         "pairwise_mean_delta_e00": float(pair_de[j].mean()),
                         "pairwise_max_delta_e00": float(pair_de[j].max()),
                         "repeat_sd_lab": repeat_sd[j].tolist(),
                         "mean_target_lab": site_targets[j].tolist()})
    report = {
        "scope": "TRAIN_ONLY_DIAGNOSTIC_NO_TRAINING_NO_RESERVED_ENDPOINTS",
        "counts": {"images": 966, "people": 24, "exact_sites": 248, "recorded_assessments": 744,
                   "triplet_pair_comparisons": 744, "unique_reference_triplets": 248,
                   "source_s4_rows_per_site": dict(Counter(len(v) for v in source_s4.values())),
                   "images_per_site": dict(sorted(image_count_by_site.items())),
                   "image_device_counts": dict(Counter(devices)),
                   "image_type_counts": dict(Counter(r["image_type"] for r in rows)),
                   "image_mode_counts": dict(Counter(r["mode"] for r in rows))},
        "provenance": {"source_sha256": source_hashes, "original_acquisition_all_sha_verified": True,
                       "role_recipe": "original skin_mskcc_data.patient_roles; not recreated GroupKFold",
                       "role_script_sha256": digest(ROOT / "scripts/skin_mskcc_data.py"),
                       "original_split_receipt_sha256": digest(prov / "split_receipt.json"),
                       "audit_script_sha256": digest(__file__),
                       "private_row_mapping_sha256": digest(output / "audit_train_rows.private.json"),
                       "S7_vs_S4_repetition_exact_match": True,
                       "numeric_decode_roles": ["train"], "old_receipts_modified": False},
        "conventions": {"instrument": "SkinColorCatch manufacturer D65 / 10-degree, original repo verified source_context.json",
                        "delta_e00": "CIEDE2000 kL=kC=kH=1 between instrument Lab values only",
                        "published_image_lab": "Only used for within-image nearest-neighbour selection; observer unknown in CSV dictionary; no cross-observer DeltaE metric computed"},
        "instrument_repeatability": {
            "assessment_interpretation": "Dictionary documents 1st/2nd/3rd colorimeter assessments; unique recordings, but timing/repositioning/random independence is not encoded in tables.",
            "pairwise_delta_e00": summary(pair_de),
            "patient_balanced": mean_ci_by_people(pair_de.ravel(), np.repeat(site_patients, 3)),
            "coordinate_absolute_pair_differences": {c: summary(np.abs(pair_diffs[:, :, k])) for k, c in enumerate(["L", "a", "b"])},
            "within_site_assessment_sample_sd": {c: summary(repeat_sd[:, k]) for k, c in enumerate(["L", "a", "b"])},
            "single_assessment_vs_triplet_mean": {"warning": "Same reading contributes to its reference, descriptive only", "delta_e00": summary(single_vs_mean)},
            "held_assessment_vs_other_two_mean": {"warning": "Assesses single-reading vs two-reading mean disagreement, not a deployable color-model error floor", "delta_e00": summary(loo_errors)},
            "triplet_target_sensitivity_to_omitting_one_assessment": {"warning": "Reference perturbation, not known latent truth", "delta_e00": summary(target_sensitivity)},
            "iid_assumption_only_coordinate_noise_variance_of_triplet_mean": {c: float(np.mean(repeat_sd[:, k] ** 2 / 3)) for k, c in enumerate(["L", "a", "b"])},
            "strata": stratum_tables},
        "site_repeatability": {"image_pair_target_delta_e00": summary(same_site_image_de),
                               "independent_across_acquisition_site_remeasurement": "NOT_IDENTIFIABLE: one reference triplet per tag_id; no independent session repeated targets",
                               "all_image_targets_are_copied_from_one_site_triplet": True,
                               "zero_pair_target_difference_interpretation": "Label reuse; not evidence that repeatability noise or biological change equals zero."},
        "oracles": oracles,
        "per_site_diagnostics": per_site,
        "limitations": ["Only 24 independent people; comparisons within sites and pairs are dependent.",
                        "Three recorded assessments estimate short-term repeat disagreement, not calibrated accuracy against an external standard.",
                        "Same-site image oracles use the very same label acquisition and are tautologically perfect.",
                        "Do not subtract DeltaE00 repeatability from color-model DeltaE00 or claim a causal error-share percentage."]}
    (output / "instrument.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"counts": report["counts"], "instrument": report["instrument_repeatability"],
                      "site_repeatability": report["site_repeatability"], "oracles": oracles}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
