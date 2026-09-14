"""Fixed-model sensitivity under bounded encoded-RGB affine contractions."""

from __future__ import annotations

import itertools

import numpy as np
from chromaseed_gated import predict
from chromaseed_kernel import coordinates, gaussian_kernel

from luma_skin_vision.color import delta_e00

ANCHORS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
DOSES = (1 / 255, 4 / 255, 16 / 255, 64 / 255)
EPS = 1e-4


def feature_direction(x, anchor):
    x = np.asarray(x, np.float64)
    a = np.broadcast_to(np.asarray(anchor, np.float64), (len(x), 3))
    d = np.zeros_like(x)
    d[:, :30] = np.tile(a, (1, 10)) - x[:, :30]
    d[:, 30:33] = -x[:, 30:33]
    return d


def basic_legal(x):
    x = np.asarray(x, np.float64)
    v = x[:, :30]
    quant = x[:, :27].reshape(-1, 9, 3)
    means, std = x[:, 27:30], x[:, 30:33]
    return (
        np.isfinite(x).all(1)
        & (v >= -1e-6).all(1)
        & (v <= 1 + 1e-6).all(1)
        & (np.diff(quant, axis=1) >= -1e-6).all((1, 2))
        & (std >= -1e-6).all(1)
        & (std**2 <= means * (1 - means) + 1e-6).all(1)
        & (np.abs(x[:, 33:]) <= 1 + 1e-6).all(1)
    )


def affine_features(x, t, anchor):
    x, t, a = np.asarray(x, np.float64), np.asarray(t, np.float64), np.asarray(anchor, np.float64)
    if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
        raise ValueError("finite color36 batch required")
    if not np.isfinite(t).all() or np.any(t < 0) or np.any(t >= 1):
        raise ValueError("0<=t<1 required")
    if a.shape not in ((3,), (len(x), 3)) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("RGB anchor in [0,1] required")
    if t.shape not in ((), (len(x),)):
        raise ValueError("scalar or one strength per row required")
    if not basic_legal(x).all():
        raise ValueError("input violates necessary feature bounds")
    out = (x + t.reshape(-1, 1) * feature_direction(x, a)).astype(np.float32)
    if not basic_legal(out).all():
        raise ValueError("affine output violates necessary feature bounds")
    return out


def grid():
    return [dict(dose=0.0, anchor_index=-1, anchor=[0.0, 0.0, 0.0])] + [
        dict(dose=t, anchor_index=i, anchor=a.tolist())
        for t in DOSES
        for i, a in enumerate(ANCHORS)
    ]


def gate_score(model, x):
    if "gate_beta" not in model:
        return None
    beta = model["gate_beta"].astype(np.float64)
    return coordinates(model, x) @ beta[1:] + beta[0]


def projection(model, x):
    z = coordinates(model, x)
    beta = model["gate_beta"].astype(np.float64)
    score = z @ beta[1:] + beta[0]
    norm2 = beta[1:] @ beta[1:]
    if norm2 <= 0:
        raise ValueError("nonconstant gate required for projection")
    point = z - score[:, None] * beta[None, 1:] / norm2
    return z, score, point, np.abs(score) / np.sqrt(norm2 * 36)


def describe(values, person, camera):
    values = np.asarray(values, np.float64)
    if len(values) == 0:
        return None
    unique = np.unique(person)
    return dict(
        person_mean=float(np.mean([values[person == p].mean() for p in unique])),
        image_mean=float(values.mean()),
        p90=float(np.quantile(values, 0.9)),
        maximum=float(values.max()),
        people=len(unique),
        rows=len(values),
        camera_person_mean={
            str(c): float(
                np.mean(
                    [
                        values[(person == p) & (camera == c)].mean()
                        for p in np.unique(person[camera == c])
                    ]
                )
            )
            for c in np.unique(camera)
        },
    )


def summaries(predictions, target, person, camera, scores):
    error = delta_e00(predictions, target[None])
    drift = delta_e00(predictions, predictions[0][None])
    transforms, doses = [], []
    flips = None if scores is None else ((scores >= 0) != (scores[0:1] >= 0))
    for i, setting in enumerate(grid()):
        transforms.append(
            dict(
                **setting,
                error=describe(error[i], person, camera),
                drift=describe(drift[i], person, camera),
                error_change=describe(error[i] - error[0], person, camera),
                sign_flip=None if flips is None else describe(flips[i], person, camera),
            )
        )
    for i, t in enumerate(DOSES):
        ix = slice(1 + 8 * i, 9 + 8 * i)
        worst_error, worst_drift = error[ix].max(0), drift[ix].max(0)
        doses.append(
            dict(
                dose=t,
                worst_error=describe(worst_error, person, camera),
                worst_drift=describe(worst_drift, person, camera),
                worst_error_change=describe(worst_error - error[0], person, camera),
                any_sign_flip=None if flips is None else describe(flips[ix].any(0), person, camera),
            )
        )
    return transforms, doses


def boundaries(models, x, person, camera):
    hard = models["hard"]
    z, score, point, distance = projection(hard, x)
    k = gaussian_kernel(point, hard["centers"], float(hard["width"]))
    base = k @ hard["coefficient"].astype(np.float64)
    correction = float(hard["rho"]) * (k @ hard["correction"].astype(np.float64))
    minus = (base - correction) * hard["y_std"] + hard["y_mean"]
    plus = (base + correction) * hard["y_std"] + hard["y_mean"]
    raw_point = point * hard["x_std"] + hard["x_mean"]
    legal = basic_legal(raw_point)
    locations, anchors, roots = [], [], []
    beta = hard["gate_beta"].astype(np.float64)
    for i, anchor in enumerate(ANCHORS):
        slope = (feature_direction(x, anchor) / hard["x_std"]) @ beta[1:]
        root = np.full(len(x), np.nan)
        np.divide(-score, slope, out=root, where=slope != 0)
        rows = np.flatnonzero((root > EPS) & (root < DOSES[-1] - EPS))
        locations.extend(rows.tolist())
        anchors.extend([i] * len(rows))
        roots.extend(root[rows].tolist())
    indices, ai, roots = (
        np.asarray(locations, np.int64),
        np.asarray(anchors, np.int64),
        np.asarray(roots),
    )
    query = x[indices]
    lo = affine_features(query, roots - EPS, ANCHORS[ai])
    hi = affine_features(query, roots + EPS, ANCHORS[ai])
    outputs = {name: np.stack((predict(m, lo), predict(m, hi))) for name, m in models.items()}
    pair_scores = np.stack((gate_score(hard, lo), gate_score(hard, hi)))
    crossed = (pair_scores[0] >= 0) != (pair_scores[1] >= 0)
    arrays = dict(
        projected_z=point,
        projected_raw=raw_point,
        distance=distance,
        minus=minus,
        plus=plus,
        basic_legal=legal,
        query_ordinal=indices,
        anchor_index=ai,
        root_t=roots,
        lower_x=lo,
        upper_x=hi,
        pair_scores=pair_scores,
        crossed=crossed,
        **{f"prediction_{key}": value for key, value in outputs.items()},
    )
    summary = dict(
        unconstrained_distance=describe(distance, person, camera),
        unconstrained_jump=describe(delta_e00(plus, minus), person, camera),
        unconstrained_basic_bounds=describe(legal, person, camera),
        legal_boundary_pairs=len(indices),
        actual_sign_flips=int(crossed.sum()),
        legal_boundary_people=len(np.unique(person[indices])),
        actual_pair_rgb_distance_bound=2 * EPS,
        jumps={
            name: describe(delta_e00(v[0], v[1]), person[indices], camera[indices])
            for name, v in outputs.items()
        },
    )
    return arrays, summary
