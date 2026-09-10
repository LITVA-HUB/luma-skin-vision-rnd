"""Evaluation uses real errors, deterministic score ties and subject resampling."""

import hashlib

import numpy as np

from luma_skin_vision.color import delta_e00


def _vectors(*arrays):
    values = [np.asarray(a) for a in arrays]
    if (
        not values
        or len(values[0]) == 0
        or any(v.ndim != 1 or len(v) != len(values[0]) for v in values)
    ):
        raise ValueError("Expected nonempty equal-length vectors")
    for v in values:
        if np.issubdtype(v.dtype, np.number) and not np.isfinite(v).all():
            raise ValueError("nonfinite evaluation input")
    return values


def selection_order(scores, ids):
    scores, ids = _vectors(scores, ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation IDs")
    # Hash tie-breaking avoids acquisition-order effects and never consults labels.
    tie = [hashlib.sha256(str(key).encode()).hexdigest() for key in ids]
    return np.lexsort((tie, scores))


def risk_coverage(errors, scores, ids, coverages=(0.6, 0.8, 0.9, 1.0), tolerance=10):
    errors, scores, ids = _vectors(errors, scores, ids)
    if np.any(errors < 0):
        raise ValueError("negative color error")
    order = selection_order(scores, ids)
    result = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0,1]")
        n = max(1, int(np.floor(coverage * len(errors) + 1e-12)))
        accepted = errors[order[:n]]
        result.append(
            {
                "requested_coverage": float(coverage),
                "coverage": n / len(errors),
                "accepted": n,
                "mean_delta_e00": float(accepted.mean()),
                "p95_delta_e00": float(np.quantile(accepted, 0.95)),
                "catastrophic_rate": float(np.mean(accepted > tolerance)),
            }
        )
    return result


def paired_bootstrap(errors_a, errors_b, subjects, *, seed=42, draws=1000):
    a, b, s = _vectors(errors_a, errors_b, subjects)
    unique = np.unique(s)
    if draws < 20 or len(unique) < 2:
        raise ValueError("Bootstrap requires >=2 subjects and >=20 draws")
    differences = np.array([(a[s == k] - b[s == k]).mean() for k in unique])
    rng = np.random.default_rng(seed)
    sampled = differences[rng.integers(0, len(unique), (draws, len(unique)))].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "subjects": len(unique),
        "estimand": "subject-macro paired mean difference A minus B",
        "draws": draws,
        "seed": seed,
    }


def bootstrap_selective(
    pred_a,
    pred_b,
    target,
    score_a,
    score_b,
    ids,
    subjects,
    coverage=0.8,
    seed=42,
    draws=300,
    image_ids=None,
):
    """Paired resample subjects, reselect equal image coverage within each resample."""
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    ea, eb = delta_e00(pred_a, target), delta_e00(pred_b, target)
    sa, sb = np.asarray(score_a), np.asarray(score_b)
    if image_ids is None:
        raise ValueError("image_ids are required for image-unit selective bootstrap")
    image_ids = np.asarray(image_ids)
    images = np.unique(image_ids)
    if len(image_ids) != len(ea):
        raise ValueError("image ID length mismatch")
    for key in images:
        if len(np.unique(subjects[image_ids == key])) != 1:
            raise ValueError("image spans multiple subjects")
    ea, eb, sa, sb, subjects = (
        np.array([ea[image_ids == key].mean() for key in images]),
        np.array([eb[image_ids == key].mean() for key in images]),
        np.array([sa[image_ids == key].max() for key in images]),
        np.array([sb[image_ids == key].max() for key in images]),
        np.array([subjects[image_ids == key][0] for key in images]),
    )
    ids = images
    if len(unique) < 2:
        return {"ci95": None, "reason": "fewer than two subjects"}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(draws):
        index, keys = [], []
        for repeat, subject in enumerate(rng.choice(unique, len(unique))):
            loc = np.flatnonzero(subjects == subject)
            index.extend(loc)
            keys.extend(f"{ids[i]}:{repeat}" for i in loc)
        ia = risk_coverage(ea[index], sa[index], keys, [coverage])[0]
        ib = risk_coverage(eb[index], sb[index], keys, [coverage])[0]
        diffs.append(ia["mean_delta_e00"] - ib["mean_delta_e00"])
    return {
        "ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
        "draws": draws,
        "seed": seed,
        "coverage": coverage,
        "estimand": "image-weighted selective risk A minus B; subject cluster bootstrap",
    }


def summarize(prediction, target, subjects, tolerance=10):
    p, y = np.asarray(prediction), np.asarray(target)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError("Expected paired Nx3 color predictions")
    errors = delta_e00(p, y)
    subjects = np.asarray(subjects).astype(str)
    if len(subjects) != len(p):
        raise ValueError("subject length mismatch")
    groups = {str(s): float(errors[subjects == s].mean()) for s in np.unique(subjects)}
    return {
        "n_region_records": len(p),
        "subjects": len(groups),
        "mean_delta_e00": float(errors.mean()),
        "median_delta_e00": float(np.median(errors)),
        "p95_delta_e00": float(np.quantile(errors, 0.95)),
        "mae_lab": np.mean(np.abs(p - y), axis=0).tolist(),
        "rmse_lab": np.sqrt(np.mean((p - y) ** 2, axis=0)).tolist(),
        "bias_lab": (p - y).mean(axis=0).tolist(),
        "subject_macro_delta_e00": float(np.mean(list(groups.values()))),
        "per_subject": groups,
        "catastrophic_tolerance": tolerance,
        "catastrophic_rate": float(np.mean(errors > tolerance)),
        "between_subject_prediction_std_lab": np.std(
            [p[subjects == s].mean(axis=0) for s in groups], axis=0
        ).tolist(),
        "between_subject_target_std_lab": np.std(
            [y[subjects == s].mean(axis=0) for s in groups], axis=0
        ).tolist(),
    }


def repeatability(repeated_labs):
    pair_errors = []
    for values in repeated_labs:
        x = np.asarray(values)
        if len(x) < 2:
            raise ValueError("at least two reference repeats required")
        pair_errors.extend(
            float(delta_e00(x[i], x[j])) for i in range(len(x)) for j in range(i + 1, len(x))
        )
    if not pair_errors:
        raise ValueError("no reference repeats")
    return {
        "reference_groups": len(repeated_labs),
        "pairs": len(pair_errors),
        "median_pair_delta_e00": float(np.median(pair_errors)),
        "p95_pair_delta_e00": float(np.quantile(pair_errors, 0.95)),
        "interpretation": "descriptive repeatability; not an instrument accuracy certificate",
    }


def reliability_bins(probability, within_tolerance, bins=5):
    p, y = _vectors(probability, within_tolerance)
    if np.any((p < 0) | (p > 1)) or not set(y).issubset({0, 1, False, True}):
        raise ValueError("invalid binary reliability data")
    groups = np.minimum((p * bins).astype(int), bins - 1)
    result = [
        {
            "count": int((groups == i).sum()),
            "mean_probability": float(p[groups == i].mean()),
            "observed_fraction": float(y[groups == i].mean()),
        }
        for i in range(bins)
        if np.any(groups == i)
    ]
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "bins": result,
        "ece_binary_tolerance_event": sum(
            b["count"] / len(p) * abs(b["mean_probability"] - b["observed_fraction"])
            for b in result
        ),
    }


def segmentation_metrics(predicted, reference):
    p, y = np.asarray(predicted, bool), np.asarray(reference, bool)
    if p.shape != y.shape:
        raise ValueError("segmentation shape mismatch")
    inter, union = np.sum(p & y), np.sum(p | y)
    return {
        "iou": float(inter / union) if union else 1.0,
        "dice": float(2 * inter / (p.sum() + y.sum())) if p.sum() + y.sum() else 1.0,
    }
