"""Independent split, weighting, and artifact checks for the TRAIN-only runner."""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _original_train_cache():
    configured = os.environ.get("LUMA_ORIGINAL_TRAIN_CACHE")
    if configured:
        return Path(configured)
    return ROOT.parents[1] / "luma-skin-vision-rnd" / "data/processed/skin_mskcc_pixels_v1/train.npz"


def test_original_train_metadata_reproduces_outer_roles_and_person_folds():
    import skin_local_search_train as runner

    cache = _original_train_cache()
    if not cache.is_file():
        pytest.skip("licensed original TRAIN cache is unavailable")
    assert cache.is_absolute()
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == runner.CACHE_HASH
    with np.load(cache, allow_pickle=False) as archive:
        person = archive["patient"]
        camera = archive["device"]

    outer = runner.roles(person, camera)
    expected = {
        "mixed": (734, 232, 18, 6, [6, 6, 6]),
        "slr_to_ipod": (323, 643, 8, 16, [3, 3, 2]),
        "ipod_to_slr": (643, 323, 16, 8, [6, 5, 5]),
    }
    for name, (fit, held) in outer.items():
        fit_images, held_images, fit_people, held_people, fold_people = expected[name]
        assert (int(fit.sum()), int(held.sum())) == (fit_images, held_images)
        assert (len(np.unique(person[fit])), len(np.unique(person[held]))) == (
            fit_people,
            held_people,
        )
        assert not np.any(fit & held)
        assert np.all(fit | held)
        assert set(person[fit]).isdisjoint(person[held])
        folds = runner.folds_for(person[fit], camera[fit])
        assert sorted(np.unique(folds).tolist()) == [0, 1, 2]
        assert [len(np.unique(person[fit][folds == k])) for k in range(3)] == fold_people
        for k in range(3):
            assert set(person[fit][folds == k]).isdisjoint(person[fit][folds != k])


def test_acquisition_weights_equalize_people_then_sites_then_images():
    import skin_local_search_train as runner

    person = np.array(["a", "a", "a", "a", "b", "b", "b", "b", "b"])
    site = np.array(["a1", "a1", "a1", "a2", "b1", "b1", "b2", "b2", "b2"])
    weights = runner.weights_for(person, site)

    assert weights.mean() == 1.0
    person_mass = [weights[person == p].sum() for p in np.unique(person)]
    np.testing.assert_allclose(person_mass, person_mass[0], rtol=0, atol=1e-14)
    for p in np.unique(person):
        site_mass = [weights[(person == p) & (site == s)].sum() for s in np.unique(site[person == p])]
        np.testing.assert_allclose(site_mass, site_mass[0], rtol=0, atol=1e-14)
        for s in np.unique(site[person == p]):
            rows = weights[(person == p) & (site == s)]
            np.testing.assert_allclose(rows, rows[0], rtol=0, atol=0)


def test_fit_selection_is_invariant_to_outer_target_sentinel(tmp_path, monkeypatch):
    import skin_local_search_train as runner

    fit = np.array([True] * 6 + [False] * 3)
    held = ~fit
    person = np.array(["p0", "p1", "p2", "p3", "p4", "p5", "h0", "h1", "h2"])
    data = {
        "color": np.arange(9 * 36, dtype=np.float64).reshape(9, 36) / 100,
        "target": np.tile(np.array([[50.0, 5.0, 10.0]]), (9, 1)),
        "patient": person,
        "site": np.array([f"s{i}" for i in range(9)]),
        "device": np.array(["SLR"] * 9),
    }

    monkeypatch.setattr(runner, "roles", lambda _person, _camera: {"probe": (fit, held)})
    monkeypatch.setattr(runner, "folds_for", lambda _person, _camera: np.arange(6) % 3)

    def fake_fit(x, y, person, site, method, parameter, seed, device, steps=512):
        assert np.max(np.abs(y)) < 1_000
        return {"method": np.asarray(method), "constant": np.asarray(y.mean(0), dtype=np.float32)}, {
            "fit_seconds": 0.0,
            "peak_allocated_bytes": 0,
            "weighted_train_normalized_sse": 0.0,
            "numeric_scalars": 3,
            "numeric_bytes": 12,
        }

    monkeypatch.setattr(runner, "fit_model", fake_fit)
    monkeypatch.setattr(
        runner,
        "predict",
        lambda model, x: np.tile(model["constant"], (len(x), 1)),
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    runner.fit_phase(data, first, "cpu")
    changed = {k: np.copy(v) for k, v in data.items()}
    changed["target"][held] = 1_000_000.0
    runner.fit_phase(changed, second, "cpu")
    assert (first / "frozen_selections.json").read_bytes() == (
        second / "frozen_selections.json"
    ).read_bytes()


def test_canonical_float32_payload_roundtrips_exactly(tmp_path):
    import skin_local_search_train as runner

    rng = np.random.default_rng(4401)
    x = rng.normal(size=(18, 36))
    y = rng.normal(size=(18, 3))
    person = np.repeat(["a", "b", "c", "d", "e", "f"], 3)
    site = np.array([f"{p}{i // 2}" for p, i in zip(person, range(18))])
    for method, parameter in (("ridge", 1.0), ("krr", 1.0), ("random_rbf", 1.0),
                              ("guided_rbf", 1.0), ("mlp", 0.001)):
        model, _ = runner.fit_model(x, y, person, site, method, parameter, 17, "cpu", steps=2)
        for value in model.values():
            if value.dtype.kind == "f":
                assert value.dtype == np.float32
        before = runner.predict(model, x)
        path = tmp_path / f"{method}.npz"
        np.savez(path, **model)
        after = runner.predict(runner.load_model(path), x)
        np.testing.assert_array_equal(after, before)
