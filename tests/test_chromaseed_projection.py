"""Projection geometry, original-control parity and portable numerical contracts."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from chromaseed_affine import fit_bank as a_bank
from chromaseed_affine import model_id as a_id
from chromaseed_affine_reference import qr_ridge
from chromaseed_kernel import gaussian_kernel
from chromaseed_projection import (
    ALPHAS,
    FAMILIES,
    REPRESENTATIONS,
    SEEDS,
    VariableColumns,
    choose,
    exact_width,
    fit_bank,
    fit_single,
    model_id,
    predict,
    project,
    projection,
)
from chromaseed_projection_numpy import Predictor


def toy(two=True):
    rng = np.random.default_rng(3871)
    # Legal color36 statistics of bounded synthetic pixels, never real participants.
    p = rng.uniform(0.05, 0.95, (42, 64, 3))
    mean, std = p.mean(1), p.std(1)
    centered = p - mean[:, None]
    corr = np.einsum("npa,npb->nab", centered, centered) / 64 / (std[:, :, None] * std[:, None, :])
    x = np.column_stack(
        [
            np.quantile(p, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], axis=1)
            .transpose(1, 0, 2)
            .reshape(42, 27),
            mean,
            std,
            corr[:, 0, 1],
            corr[:, 0, 2],
            corr[:, 1, 2],
        ]
    ).astype(np.float32)
    y = rng.uniform([35, 2, 5], [75, 18, 30], (42, 3))
    person = np.repeat(np.arange(14), 3)
    site = np.tile([0, 0, 1], 14)
    camera = np.where(person < (7 if two else 14), "SLR", "ipod")
    return x, y, person, site, camera


@pytest.fixture(scope="module")
def bank():
    return fit_bank(*toy(), rank=4)


def test_weighted_covariance_projection_matches_svd_and_weight_rescaling():
    rng = np.random.default_rng(451)
    z, w = rng.normal(size=(83, 36)), rng.uniform(0.1, 2, 83)
    m = np.average(z, axis=0, weights=w)
    _, s, vt = np.linalg.svd((z - m) * np.sqrt(w / w.sum())[:, None], full_matrices=False)
    v = vt.T
    v *= np.where(v[np.argmax(np.abs(v), axis=0), np.arange(36)] < 0, -1, 1)
    expected = v[:, :8] / np.sqrt(0.9 * s[:8] ** 2 + 0.1 * np.mean(s**2))
    payload, info = projection(z, w, 8, 0.1)
    np.testing.assert_allclose(payload["projection"], expected, atol=2e-7, rtol=1e-6)
    np.testing.assert_array_equal(payload["projection_mean"], m.astype(np.float32))
    same, _ = projection(z, 7 * w, 8, 0.1)
    np.testing.assert_allclose(same["projection"], payload["projection"], atol=2e-7, rtol=1e-6)
    assert info["dimension"] == 8 and info["shrinkage"] == 0.1


def test_unshrunk_full_projection_whitens_and_shrunk_full_preserves_distances():
    z = np.random.default_rng(16).normal(size=(140, 36)) * np.linspace(0.1, 4, 36)
    w = np.linspace(0.5, 2, len(z))
    m = np.average(z, axis=0, weights=w)
    c = (z - m).T @ (w[:, None] * (z - m)) / w.sum()
    pure, _ = projection(z, w, 36, 0)
    p = pure["projection"].astype(np.float64)
    np.testing.assert_allclose(p.T @ c @ p, np.eye(36), atol=1e-5, rtol=0)
    flat, _ = projection(z, w, 36, 1)
    p = flat["projection"].astype(np.float64)
    np.testing.assert_allclose(p @ p.T, np.eye(36) / (np.trace(c) / 36), atol=1e-8, rtol=1e-6)


def test_constant_covariance_is_finite_and_invalid_projections_rejected():
    for d in (8, 16, 36):
        payload, info = projection(np.zeros((40, 36)), np.ones(40), d, 0)
        assert np.isfinite(payload["projection"]).all() and info["floored_directions"] == d
    for d, t in ((0, 0.1), (37, 0.1), (8, -0.1), (8, 1.1)):
        with pytest.raises(ValueError):
            projection(np.zeros((40, 36)), np.ones(40), d, t)


@pytest.mark.parametrize("d", [8, 16, 36])
def test_variable_width_and_columns_match_exhaustive_distances(d):
    x = np.random.default_rng(d).normal(size=(37, d))
    x[1] = x[0]
    distances = np.array(
        [np.sqrt(np.mean((x[i] - x[j]) ** 2)) for i in range(len(x)) for j in range(i + 1, len(x))]
    )
    positive = np.sort(distances[distances > 1e-10])
    expected = positive[(len(positive) - 1) // 2]
    width, info = exact_width(x)
    np.testing.assert_allclose(width, expected, atol=1e-13, rtol=0)
    op = VariableColumns(x, width)
    k = gaussian_kernel(x, x, width)
    for i in (0, 1, 17, 36):
        np.testing.assert_allclose(op.column(i), k[:, i], atol=2e-14, rtol=0)
    assert info["pair_count"] == 666 and op.entries_evaluated == 4 * len(x)


def test_raw_identity_exactly_preserves_all_a_zero_augmentation_payloads(bank):
    models, receipt = bank
    previous, _ = a_bank(*toy(), rank=4)
    assert len(models) == 468 and receipt["new_coefficient_solutions"] == 468
    for family in FAMILIES:
        for seed in SEEDS:
            for alpha in ALPHAS:
                m, old = (
                    models[model_id(family, seed, "raw", alpha)],
                    previous[a_id(family, seed, alpha, 0)],
                )
                assert set(m) == set(old)
                for key in m:
                    np.testing.assert_array_equal(m[key], old[key])


def test_projected_selected_fit_and_bank_export_are_exact_and_target_free_projection(bank):
    models, _ = bank
    data = toy()
    for family in FAMILIES:
        m, _ = fit_single(*data, family, 17, "d8_t01", 1.0, rank=4)
        saved = models[model_id(family, 17, "d8_t01", 1.0)]
        for key in m:
            np.testing.assert_array_equal(m[key], saved[key])
    changed = (data[0], data[1] + np.linspace(0, 20, len(data[1]))[:, None], *data[2:])
    m, _ = fit_single(*changed, "norm_static", 17, "d8_t01", 1.0, rank=4)
    for key in ("projection", "projection_mean", "centers", "width"):
        np.testing.assert_array_equal(
            m[key], models[model_id("norm_static", 17, "d8_t01", 1.0)][key]
        )


def test_single_camera_joint_payloads_are_exact_static_aliases():
    models, receipt = fit_bank(*toy(False), rank=4)
    assert receipt["gate_fits"] == 0 and receipt["new_coefficient_solutions"] == 234
    for family in ("norm_joint_soft", "perceptual_joint_soft"):
        for representation in REPRESENTATIONS:
            a = models[model_id(family, 17, representation["name"], 1.0)]
            b = models[
                model_id(family.replace("joint_soft", "static"), 17, representation["name"], 1.0)
            ]
            assert set(a) == set(b)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])


def test_projected_normalized_readout_matches_augmented_qr_solution(bank):
    from chromaseed_perceptual_audit import balanced

    models, _ = bank
    m = models[model_id("norm_static", 17, "d8_t01", 0.1)]
    x, y, p, s, _ = toy()
    q = project(m, x)
    k = gaussian_kernel(q, m["centers"], float(m["width"]))
    km = gaussian_kernel(m["centers"], m["centers"], float(m["width"]))
    u, v, _ = np.linalg.svd(km)
    keep = v > v.max() * 1e-8
    white = u[:, keep] / np.sqrt(v[keep])
    theta = qr_ridge(k @ white, (y - m["y_mean"]) / m["y_std"], balanced(p, s), np.eye(3), 0.1)
    expected = (k @ white @ theta) * m["y_std"] + m["y_mean"]
    np.testing.assert_allclose(predict(m, x), expected, atol=1e-4, rtol=0)


def test_portable_predictor_matches_serialized_all_representations_and_modes(bank):
    models, _ = bank
    x = toy()[0]
    for representation in REPRESENTATIONS:
        for family in ("perceptual_static", "perceptual_joint_soft"):
            m = models[model_id(family, 17, representation["name"], 0.1)]
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **m)
            buffer.seek(0)
            with np.load(buffer, allow_pickle=False) as archive:
                call = Predictor({k: archive[k] for k in archive.files})
            np.testing.assert_allclose(
                np.array([call(v) for v in x]), predict(m, x), atol=2e-8, rtol=0
            )
            assert call.cached_array_bytes > sum(v.nbytes for v in m.values())
    m = models[model_id("norm_static", 17, "d8_t01", 0.1)]
    for corrupt in (
        {**m, "projection": m["projection"].astype(np.float64)},
        {k: v for k, v in m.items() if k != "projection_mean"},
        {**m, "width": np.array(-1, np.float32)},
    ):
        with pytest.raises(ValueError):
            Predictor(corrupt)
    with pytest.raises(ValueError):
        Predictor(m)(np.zeros(35))


def test_compact_policy_requires_both_error_guards_and_has_raw_fallback():
    candidates = [
        dict(representation="raw", order=0, alpha=0.1, clean=5.0, p90=10.0, numeric_bytes=20284),
        dict(
            representation="d8_t01", order=2, alpha=0.1, clean=5.04, p90=10.09, numeric_bytes=7244
        ),
        dict(
            representation="d8_t05", order=3, alpha=1.0, clean=5.01, p90=10.11, numeric_bytes=7244
        ),
        dict(representation="d16_t01", order=6, alpha=0.1, clean=4.9, p90=9.9, numeric_bytes=12492),
    ]
    assert choose(candidates, "quality")["representation"] == "d16_t01"
    assert choose(candidates, "compact")["representation"] == "d8_t01"
    candidates[1]["clean"] = 5.051
    assert choose(candidates, "compact")["representation"] == "d16_t01"
    assert choose(candidates[:1], "compact")["representation"] == "raw"
    with pytest.raises(ValueError):
        choose(candidates, "unknown")
