import numpy as np
import pytest

from luma_skin_vision.cc.core import (
    angular,
    experts,
    linearize,
    reproduction,
    selective_curve,
    summarize,
)


def test_angular_scale_invariance_and_orthogonal():
    assert angular([[1, 0, 0]], [[0, 1, 0]])[0] == pytest.approx(90)
    assert angular([[1, 2, 3]], [[2, 4, 6]])[0] < 1e-5
    with pytest.raises(ValueError):
        angular([[0, 0, 0]], [[1, 1, 1]])


def test_reproduction_is_corrected_neutral_not_recovery():
    gt = np.array([[1.0, 2, 4]])
    pred = np.array([[2.0, 2, 2]])
    expected = np.degrees(np.arccos(3.5 / (np.sqrt(3) * np.sqrt(5.25))))
    assert reproduction(pred, gt)[0] == pytest.approx(expected)
    assert reproduction(gt * 3, gt)[0] < 1e-5


def test_summary_and_coverage_are_actual_accepted_counts():
    result = summarize(np.arange(1, 101))
    assert result["median"] == 50.5
    assert result["best25"] == 13
    assert result["worst25"] == 88
    curve = selective_curve(np.arange(1, 11), np.arange(1, 11), [str(i) for i in range(10)])
    assert curve["fixed"]["80"]["n"] == 8
    assert curve["fixed"]["80"]["mean"] == 4.5
    assert curve["fixed"]["95"]["coverage"] == 0.9


def test_linear_loader_subtracts_black_not_srgb_gamma():
    image = np.full((10, 10, 3), [3048, 4048, 6048], dtype=np.uint16)
    x = linearize(image, black=2048, white=15000)
    np.testing.assert_allclose(x[0, 0], np.array([1000, 2000, 4000]) / (15000 - 2048))
    est = experts(x)
    for e in est[:3]:
        assert angular(e[None], [[1, 2, 4]])[0] < 1e-5
    assert np.isfinite(est).all()
    with pytest.raises(ValueError):
        linearize(image.astype(np.uint8))


def test_model_contract_and_finite_gradient():
    torch = pytest.importorskip("torch")
    from luma_skin_vision.cc.model import CompactCC, reproduction_loss

    torch.set_num_threads(2)
    for mixture in [False, True]:
        model = CompactCC(mixture).eval()
        pred, context = model(torch.rand(2, 3, 64, 64), torch.ones(2, 4, 3) / np.sqrt(3))
        assert pred.shape == (2, 3) and context.shape == (2, 64)
        assert (pred > 0).all()
        loss = reproduction_loss(pred, torch.ones(2, 3))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_capture_day_partition_is_stable():
    from luma_skin_vision.cc.data import partition

    assert partition("2020:06:22") == partition("2020:06:22")
    assert partition("2020:06:22") in {"train", "val", "risk", "cal"}


def test_risk_fitting_cannot_see_test_errors():
    from luma_skin_vision.cc.benchmark import apply_risk, fit_risk

    rng = np.random.default_rng(1)
    f = rng.normal(size=(60, 7))
    e = rng.uniform(0, 20, 60)
    a = fit_risk(f, e, np.arange(30), np.arange(30, 40))
    changed = e.copy()
    changed[40:] = 180
    b = fit_risk(f, changed, np.arange(30), np.arange(30, 40))
    np.testing.assert_array_equal(apply_risk(f, a), apply_risk(f, b))


def test_data_cache_fingerprint_detects_target_change(tmp_path):
    from luma_skin_vision.cc.benchmark import data_hashes

    for name in ["cube.npz", "cube_manifest.json", "sony.npz", "sony_manifest.json"]:
        (tmp_path / name).write_bytes(b"original")
    original = data_hashes(tmp_path)
    (tmp_path / "cube.npz").write_bytes(b"changed GT")
    assert data_hashes(tmp_path) != original
