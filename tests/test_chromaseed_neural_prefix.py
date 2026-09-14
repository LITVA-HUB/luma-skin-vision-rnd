"""Behavioral tests for exact prefixes, fixed schedules and conditional selection."""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def mod():
    assert importlib.util.find_spec("chromaseed_neural_prefix_numpy"), "missing NP implementation"
    return importlib.import_module("chromaseed_neural_prefix_numpy")


def parent(family):
    k, d, h = {
        "plain": (1, 36, 69),
        "local2": (2, 39, 32),
        "local4": (4, 39, 16),
        "blind4": (4, 39, 16),
        "e2e4": (4, 39, 16),
    }[family]
    rng = np.random.default_rng(83)
    return dict(
        family=np.array(family),
        theta=rng.normal(0, 0.05, (k, (d + 1) * h + (h + 1) * 3)).astype(np.float32),
        x_mean=np.zeros(36, np.float32),
        x_std=np.ones(36, np.float32),
        y_mean=np.array([50, 12, 20], np.float32),
        y_std=np.ones(3, np.float32),
    )


@pytest.mark.parametrize("family", ["plain", "local2", "local4", "blind4", "e2e4"])
def test_every_prefix_matches_original_clean_head(family):
    m = mod()
    from chromaseed_local_denoise_audit import direct

    p = parent(family)
    x = np.random.default_rng(34).normal(size=(11, 36)).astype(np.float32)
    expected = direct(p, x)
    for j in range(1, expected.shape[1] + 1):
        export = m.export_prefix(p, j)
        np.testing.assert_allclose(m.predict(export, x), expected[:, j - 1], rtol=0, atol=2e-8)
        single = m.Predictor(export)
        np.testing.assert_allclose(
            np.stack([single(row) for row in x]), expected[:, j - 1], rtol=0, atol=2e-8
        )
        assert int(export["original_k"]) == len(p["theta"])


def test_prefix_two_keeps_original_four_stage_schedule_and_returns_clean():
    m = mod()
    p = parent("local4")
    p["theta"][:] = 0
    p["y_mean"][:] = 0
    p["theta"][0, -3:] = 1
    # Block2 copies its three nonnegative state coordinates through three ReLU units.
    w = p["theta"][1, : 39 * 16].reshape(39, 16)
    w[36:39, :3] = np.eye(3)
    v = p["theta"][1, 40 * 16 : -3].reshape(16, 3)
    v[:3] = np.eye(3)
    result = m.Predictor(m.export_prefix(p, 2))(np.zeros(36, np.float32))
    np.testing.assert_allclose(result, np.full(3, np.sin(np.pi / 8)), rtol=0, atol=1e-12)
    assert not np.allclose(result, np.sin(np.pi / 4))


def test_blind_head_collapses_and_ignores_earlier_heads_and_state_weights():
    m = mod()
    p = parent("blind4")
    out = m.export_prefix(p, 3)
    assert len([key for key in out if key.startswith("w")]) == 1
    assert out["w0"].shape == (36, 16)
    p["theta"][:2] += 999
    p["theta"][2, : 39 * 16].reshape(39, 16)[36:] += 999
    changed = m.export_prefix(p, 3)
    for key in out:
        np.testing.assert_array_equal(out[key], changed[key])


def test_payload_size_counts_only_retained_weights_and_metadata():
    m = mod()
    first = m.export_prefix(parent("e2e4"), 1)
    full = m.export_prefix(parent("e2e4"), 4)
    assert m.capacity(first) == dict(parameters=643, numeric_bytes=2886, executed_blocks=1)
    assert m.capacity(full) == dict(parameters=2716, numeric_bytes=11178, executed_blocks=4)
    assert first["w0"].dtype == np.float32


@pytest.mark.parametrize("bad", [0, 5, 1.5, True])
def test_invalid_export_prefix_rejected(bad):
    with pytest.raises(ValueError):
        mod().export_prefix(parent("local4"), bad)


@pytest.mark.parametrize("bad", [np.zeros((1, 36)), np.zeros(35), np.full(36, np.nan)])
def test_single_consumer_rejects_invalid_input(bad):
    m = mod()
    with pytest.raises(ValueError):
        m.Predictor(m.export_prefix(parent("local4"), 1))(bad)


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_k", np.array(2, np.uint8)),
        ("prefix", np.array(1.5)),
        ("w0", np.zeros((39, 16), np.float32)),
        ("y_std", np.zeros(3, np.float32)),
    ],
)
def test_malformed_payload_rejected(key, value):
    m = mod()
    p = m.export_prefix(parent("local4"), 1)
    p[key] = value
    with pytest.raises(ValueError):
        m.Predictor(p)


def test_compact_policy_obeys_error_tolerance_and_tie_order():
    m = mod()
    candidates = [
        dict(prefix=1, clean=5.11, p90=8.0, numeric_bytes=2886, executed_blocks=1),
        dict(prefix=2, clean=5.09, p90=8.0, numeric_bytes=5650, executed_blocks=2),
        dict(prefix=3, clean=5.0, p90=9.0, numeric_bytes=8414, executed_blocks=3),
        dict(prefix=4, clean=5.0, p90=8.0, numeric_bytes=11178, executed_blocks=4),
    ]
    chosen = m.choose(candidates)
    assert chosen["quality"]["prefix"] == 4
    assert chosen["compact"]["prefix"] == 2
    candidates[0]["clean"] = 5.1
    assert m.choose(candidates)["compact"]["prefix"] == 1
