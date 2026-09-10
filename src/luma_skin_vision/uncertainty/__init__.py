"""Small residual regressor fit exclusively to subject-out-of-fold errors."""

import numpy as np


def error_features(method, predicted_lab, ambiguity):
    if method == "proposed_v1":
        return np.column_stack([predicted_lab, ambiguity])
    if method == "baseline_c_plus":
        return np.column_stack([predicted_lab, ambiguity[:, 6:]])
    return np.ones((len(predicted_lab), 1))


def fit_error(features, residuals, *, provenance, ridge=1.0):
    if provenance != "subject_out_of_fold":
        raise ValueError("Error model requires subject-out-of-fold residuals")
    x, y = np.asarray(features), np.asarray(residuals)
    if (
        x.ndim != 2
        or len(x) != len(y)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(y < 0)
    ):
        raise ValueError("invalid residual training data")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-5)
    z = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    penalty = np.eye(z.shape[1]) * ridge
    penalty[-1, -1] = 0
    coefficient = np.linalg.solve(z.T @ z + penalty, z.T @ np.log1p(y))
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "provenance": provenance,
        "output": "estimated_delta_e00_uncalibrated",
    }


def predict_error(model, features):
    x = (np.asarray(features) - model["mean"]) / model["scale"]
    prediction = np.column_stack([x, np.ones(len(x))]) @ np.asarray(model["coefficient"])
    return np.maximum(0, np.expm1(np.clip(prediction, -10, 10)))
