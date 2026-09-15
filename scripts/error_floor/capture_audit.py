"""TRAIN-only within-site image variability audit; no fitting or face parsing.

Usage: PYTHONPATH=src python scripts/error_floor/capture_audit.py \
  --rows ../private_artifacts/error_floor/audit_train_rows.private.json \
  --output ../private_artifacts/error_floor/capture

Original photos are deliberately never copied into the output. Feature-level
private files contain source identifiers; aggregate files use local aliases.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps
from scipy.stats import pearsonr, spearmanr

from luma_skin_vision.color import delta_e00, srgb_to_lab, srgb_to_linear

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    v = np.asarray(values, dtype=float).ravel()
    assert np.isfinite(v).all()
    if not len(v):
        return {"n": 0, "mean": None, "median": None, "p90": None,
                "p95": None, "max": None}
    return {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.quantile(v, .9)), "p95": float(np.quantile(v, .95)),
            "max": float(v.max())}


def color36(rgb):
    # Quantile-major RGB, mean, population std and correlations RG/RB/GB.
    z = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.
    mean, std = z.mean(0), z.std(0)
    constant = np.ptp(z, axis=0) == 0
    std[constant] = 0
    covariance = (z-mean).T @ (z-mean) / len(z)
    corr = covariance / np.maximum(std[:, None] * std[None, :], 1e-12)
    corr[constant, :] = 0
    corr[:, constant] = 0
    return np.concatenate([
        np.quantile(z, [.01, .05, .1, .25, .5, .75, .9, .95, .99], axis=0).ravel(),
        mean, std, np.clip(corr, -1, 1)[np.triu_indices(3, 1)]])


def hist_mean_median(image):
    h = np.array(image.histogram(), dtype=np.int64).reshape(3, 256)
    n = image.width * image.height
    assert (h.sum(1) == n).all()
    mean = h @ np.arange(256) / n / 255.
    cumul = h.cumsum(1)
    # Exact conventional median, including averaging the middle two for even n.
    median = np.array([(np.searchsorted(c, (n-1)//2+1) +
                        np.searchsorted(c, n//2+1)) / 2 for c in cumul]) / 255.
    return mean, median


def decode_one(row, images):
    path = images / (row["image"] + ".jpg")
    source_sha = sha(path)
    with Image.open(path) as source:
        orientation = int(source.getexif().get(274, 1))
        original_size = source.size
        im = ImageOps.exif_transpose(source)
        icc = im.info.get("icc_profile")
        if icc:
            im = ImageCms.profileToProfile(
                im, ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                ImageCms.createProfile("sRGB"), outputMode="RGB")
        else:
            if im.mode not in ("RGB", "RGBA", "L"):
                raise ValueError("Unprofiled non-RGB input unsupported")
            im = im.convert("RGB")
        w, h = im.size
        side = int(.8 * min(w, h))
        left, top = (w-side)//2, (h-side)//2
        crop = im.crop((left, top, left+side, top+side))
        mean, median = hist_mean_median(crop)
        rgb128 = np.asarray(crop.resize((128, 128), Image.Resampling.LANCZOS))
        crop_array = np.asarray(crop)
        original_clipping = [
            float(np.mean(np.any(crop_array >= 253, axis=-1))),
            float(np.mean(np.any(crop_array <= 2, axis=-1)))]
    flat = rgb128.reshape(-1, 3).astype(np.float64) / 255.
    brightness = flat.mean(1)
    luminance = srgb_to_linear(flat) @ np.array([.2126729, .7151522, .0721750])
    features = color36(rgb128)
    mean128, median128 = flat.mean(0), np.median(flat, axis=0)
    diagnostic = np.array([
        brightness.mean(), brightness.std(), luminance.mean(),
        np.mean(np.any(flat >= .99, axis=1)),
        np.mean(np.any(flat <= .01, axis=1)),
        np.mean(np.any((flat >= .99) | (flat <= .01), axis=1)),
        *original_clipping, float(np.var(flat, axis=0).mean())])
    return {"source_sha256": source_sha, "color36": features,
            "original_mean_rgb": mean, "original_median_rgb": median,
            "original_mean_lab": srgb_to_lab(mean),
            "original_median_lab": srgb_to_lab(median),
            "mean128_lab": srgb_to_lab(mean128), "median128_lab": srgb_to_lab(median128),
            "diagnostic": diagnostic,
            "width": w, "height": h, "original_width": original_size[0],
            "original_height": original_size[1], "crop_xyxy": [left, top, left+side, top+side],
            "icc_sha256": hashlib.sha256(icc).hexdigest() if icc else "absent_assumed_sRGB",
            "exif_orientation": orientation}


def correlations(left, right):
    x, y = np.asarray(left), np.asarray(right)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n_sites": len(x), "pearson": None, "spearman": None}
    return {"n_sites": len(x), "pearson": float(pearsonr(x, y).statistic),
            "spearman": float(spearmanr(x, y).statistic),
            "scope": "descriptive_unique_sites_not_independent_people"}


def main(args):
    start = time.perf_counter()
    rows_all = json.loads(args.rows.read_bytes())
    assert len(rows_all) == 966 and len({r["patient"] for r in rows_all}) == 24
    assert len({r["site"] for r in rows_all}) == 248
    assert len({r["image"] for r in rows_all}) == 966
    assert len({(r["patient"], r["site"]) for r in rows_all}) == 248
    missing = [r for r in rows_all if not (args.images/(r["image"]+".jpg")).exists()]
    if missing and not args.allow_partial:
        raise RuntimeError(f"{len(missing)}/966 TRAIN photographs missing; no complete audit emitted")
    rows = [r for r in rows_all if (args.images/(r["image"]+".jpg")).exists()]
    if not rows:
        raise RuntimeError("No TRAIN photographs available")
    args.output.mkdir(exist_ok=True, parents=True)
    values = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, value in enumerate(pool.map(lambda r: decode_one(r, args.images), rows)):
            values.append(value)
            if (i+1) % 50 == 0:
                print(json.dumps({"decoded": i+1, "total": len(rows),
                                  "seconds": time.perf_counter()-start}), flush=True)
    data = {k: np.stack([v[k] for v in values]) for k in values[0]}
    for k in ("image", "patient", "site", "device", "mode", "image_type", "anatomic_site"):
        data[k] = np.array([r[k] for r in rows])
    row_index = {r["image"]: i for i, r in enumerate(rows_all)}
    data["row_index"] = np.array([row_index[r["image"]] for r in rows], dtype=np.int64)
    data["target"] = np.array([r["target"] for r in rows])
    data["repetitions"] = np.array([r["repetitions"] for r in rows])
    data["diagnostic_names"] = np.array([
        "brightness_mean_encoded_RGB", "brightness_sd_encoded_RGB", "linear_Y_mean",
        "resized_high_clip_pixel_fraction", "resized_low_clip_pixel_fraction",
        "resized_any_clip_pixel_fraction", "original_high_clip_pixel_fraction",
        "original_low_clip_pixel_fraction", "resized_channel_variance_mean"])
    np.savez_compressed(args.output/"capture_features.private.npz", **data)

    # No endpoint or model normalizer is fitted. Descriptive scaling uses only this
    # TRAIN snapshot. Partial-snapshot output is never presented as the full cohort.
    standard_scale = data["color36"].std(0)
    standard_scale[standard_scale < 1e-12] = 1.
    grouped = defaultdict(list)
    for i, r in enumerate(rows):
        grouped[(r["patient"], r["site"])].append(i)
    p_alias = {p: f"P{i+1:02d}" for i, p in enumerate(sorted({r["patient"] for r in rows_all}))}
    s_alias = {s: f"S{i+1:03d}" for i, s in enumerate(sorted({r["site"] for r in rows_all}))}
    anonymized = {k: v for k, v in data.items()
                  if k not in {"image", "patient", "site", "source_sha256", "icc_sha256"}}
    anonymized.update(
        image=np.array([f"I{i+1:04d}" for i in data["row_index"]]),
        patient=np.array([p_alias[r["patient"]] for r in rows]),
        site=np.array([s_alias[r["site"]] for r in rows]),
        icc_profile_present=data["icc_sha256"] != "absent_assumed_sRGB")
    np.savez_compressed(args.output/"capture_features_anonymized.npz", **anonymized)
    per_site, pairs = [], []
    pair_quantities = ["original_central_mean_delta_e00", "original_central_median_delta_e00",
                       "resized_central_mean_delta_e00", "resized_central_median_delta_e00",
                       "color36_raw_rms", "color36_train_z_rms", "instrument_target_delta_e00"]
    for (patient, site), indices in sorted(grouped.items()):
        sitepairs = []
        reps = data["repetitions"][indices[0]]
        reference_pairs = np.array([delta_e00(reps[i], reps[j]) for i, j in itertools.combinations(range(3), 2)])
        for i, j in itertools.combinations(indices, 2):
            assert np.array_equal(data["target"][i], data["target"][j]), "Reference acquisition semantics changed"
            a, b = rows[i], rows[j]
            delta = data["color36"][i] - data["color36"][j]
            pair = {
                "person_id": p_alias[patient], "site_id": s_alias[site],
                "device": a["device"] if a["device"] == b["device"] else "cross_device",
                "same_device": a["device"] == b["device"], "same_mode": a["mode"] == b["mode"],
                "same_image_type": a["image_type"] == b["image_type"],
                "mode_pair": " / ".join(sorted([a["mode"], b["mode"]])),
                "image_type_pair": " / ".join(sorted([a["image_type"], b["image_type"]])),
                "anatomic_site": a["anatomic_site"],
                "original_central_mean_delta_e00": float(delta_e00(data["original_mean_lab"][i], data["original_mean_lab"][j])),
                "original_central_median_delta_e00": float(delta_e00(data["original_median_lab"][i], data["original_median_lab"][j])),
                "resized_central_mean_delta_e00": float(delta_e00(data["mean128_lab"][i], data["mean128_lab"][j])),
                "resized_central_median_delta_e00": float(delta_e00(data["median128_lab"][i], data["median128_lab"][j])),
                "color36_raw_rms": float(np.sqrt(np.mean(delta**2))),
                "color36_train_z_rms": float(np.sqrt(np.mean((delta/standard_scale)**2))),
                "instrument_target_delta_e00": float(delta_e00(data["target"][i], data["target"][j]))}
            sitepairs.append(pair)
            pairs.append(pair)
        per_site.append({"site_id": s_alias[site], "person_id": p_alias[patient],
                         "images": len(indices), "pairs": len(sitepairs),
                         "device": rows[indices[0]]["device"],
                         "anatomic_site": rows[indices[0]]["anatomic_site"],
                         "original_roi_mean_lab_sample_sd": data["original_mean_lab"][indices].std(0, ddof=1).tolist() if len(indices) > 1 else None,
                         "original_roi_median_lab_sample_sd": data["original_median_lab"][indices].std(0, ddof=1).tolist() if len(indices) > 1 else None,
                         "color36_sample_sd": data["color36"][indices].std(0, ddof=1).tolist() if len(indices) > 1 else None,
                         "brightness_range_encoded_RGB": float(np.ptp(data["diagnostic"][indices, 0])),
                         "max_resized_any_clip_pixel_fraction": float(data["diagnostic"][indices, 5].max()),
                         "instrument_pairwise_mean_delta_e00": float(reference_pairs.mean()),
                         "instrument_pairwise_max_delta_e00": float(reference_pairs.max()),
                         **{k: summary([p[k] for p in sitepairs]) for k in pair_quantities}})
    def table(subset):
        return {"pairs": len(subset), "sites": len({p["site_id"] for p in subset}),
                "people": len({p["person_id"] for p in subset}),
                **{k: summary([p[k] for p in subset]) for k in pair_quantities}}
    by_person = defaultdict(list)
    for pair in pairs:
        by_person[pair["person_id"]].append(pair)
    person_mean_summary = {
        k: summary([np.mean([p[k] for p in subset]) for subset in by_person.values()])
        for k in pair_quantities}
    strata = {key: [{"group": str(value), **table([p for p in pairs if p[key] == value])}
                    for value in sorted({p[key] for p in pairs})]
              for key in ("device", "mode_pair", "image_type_pair", "same_image_type", "anatomic_site")}
    eligible_sites = [s for s in per_site if s["pairs"]]
    correlations_table = {
        key: correlations([s["instrument_pairwise_mean_delta_e00"] for s in eligible_sites],
                          [s[key]["mean"] for s in eligible_sites])
        for key in pair_quantities if key != "instrument_target_delta_e00"}
    results = {
        "scope": "PARTIAL_TRAIN_DEBUG_ONLY" if missing else "COMPLETE_TRAIN_CAPTURE_DIAGNOSTIC_NO_TRAINING",
        "counts": {"images": len(rows), "missing_images": len(missing),
                   "people": len({r["patient"] for r in rows}), "exact_sites": len(grouped),
                   "same_site_pairs": len(pairs), "same_mode_pairs": sum(p["same_mode"] for p in pairs),
                   "cross_device_pairs": sum(not p["same_device"] for p in pairs),
                   "same_type_pairs": sum(p["same_image_type"] for p in pairs),
                   "cross_type_pairs": sum(not p["same_image_type"] for p in pairs),
                   "icc_profiles": dict(Counter(data["icc_sha256"].tolist())),
                   "exif_orientation": dict(Counter(data["exif_orientation"].tolist())),
                   "original_resolutions": {f"{w}x{h}": n for (w, h), n in Counter(zip(data["original_width"].tolist(), data["original_height"].tolist())).items()}},
        "contract": {
            "decode": "EXIF transpose; embedded ICC to sRGB via Pillow ImageCms; unprofiled RGB assumes sRGB",
            "roi": "center square int(0.8*min(width,height)); floor-centered; no verified skin mask",
            "original_roi": "exact RGB histogram mean and median before resize; encoded sRGB statistics",
            "representation": "same 36-feature layout: 27 quantiles, 3means, 3population std, RG/RB/GB correlations;128x128 Lanczos",
            "original_C_cache_identity": "not asserted: original C cache unavailable; historical code/contract reconstructed",
            "observed_lab_convention": "sRGB IEC transfer + D65 [0.95047,1,1.08883], CIE1931 2deg",
            "reference_lab_convention": "MSKCC D65/CIE1964 10deg; no direct cross-observer observed-v-reference DeltaE computed",
            "median_skin_color_limitation": "ROI median is a robust pixel statistic, not verified skin-only color or instrument-site segmentation",
            "color36_raw_rms_units": "mixed feature-coordinate RMS; not perceptual color units",
            "color36_train_z_rms": "RMS of pair differences divided by feature population SD over available TRAIN images; descriptive only",
            "within_site_target_warning": "all image labels at each site repeat the same recorded reference triplet; target invariance does not demonstrate temporal instrument stability",
            "capture_warning": "mode differences are capture transformations, not independent repeats under identical capture conditions; device is person-confounded",
            "prohibited_interpretation": "observed within-site DeltaE is not model error, irreducible noise floor, or a term subtractable from OOF DeltaE"},
        "all_pairs": table(pairs), "strata": strata,
        "site_equal_weight": {k: summary([s[k]["mean"] for s in eligible_sites]) for k in pair_quantities},
        "person_equal_weight_mean_of_each_person_pair_means": person_mean_summary,
        "instrument_vs_image_variability_site_correlations": correlations_table,
        "per_site": per_site,
        "image_diagnostics": {name: summary(data["diagnostic"][:, i]) for i, name in enumerate(data["diagnostic_names"])},
        "source_rows_sha256": sha(args.rows), "script_sha256": sha(__file__),
        "private_features_sha256": sha(args.output/"capture_features.private.npz"),
        "anonymized_features_sha256": sha(args.output/"capture_features_anonymized.npz"),
        "elapsed_seconds": time.perf_counter()-start}
    (args.output/"capture.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    (args.output/"capture_pairs_anonymized.json").write_text(json.dumps(pairs, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"scope": results["scope"], "counts": results["counts"],
                      "all_pairs": results["all_pairs"], "seconds": time.perf_counter()-start}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=ROOT/"data/public/mskcc_skin_v1/images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    main(parser.parse_args())
