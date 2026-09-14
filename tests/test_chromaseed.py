"""Independent numerical and provenance checks for synthetic palette transfer."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_color36_matches_independent_quantiles_and_correlations():
    from chromaseed import color36

    rng = np.random.default_rng(883)
    samples = rng.integers(0, 256, size=(4, 143, 3)).astype(float) / 255
    samples[0, :, 0] = .4
    samples[1] = .25
    result = color36(samples)
    expected = []
    for pixels in samples:
        std = pixels.std(0)
        constant = np.ptp(pixels, axis=0) == 0
        std[constant] = 0
        corr = []
        for a, b in ((0, 1), (0, 2), (1, 2)):
            corr.append(0 if constant[a] or constant[b] else np.corrcoef(pixels[:, a], pixels[:, b])[0, 1])
        expected.append(np.r_[np.quantile(pixels, [.01, .05, .1, .25, .5, .75, .9, .95, .99], axis=0).ravel(), pixels.mean(0), std, corr])
    np.testing.assert_allclose(result, expected, atol=2e-14, rtol=0)


def test_canonical_palette_has_exact_colorimetric_targets_and_gamut():
    from chromaseed import canonical_palette

    from luma_skin_vision.color import srgb_to_lab

    rgb, target = canonical_palette(700, 819)
    assert rgb.shape == target.shape == (700, 3)
    assert np.all((rgb >= .02) & (rgb <= .98))
    np.testing.assert_allclose(srgb_to_lab(rgb), target, atol=1e-10, rtol=0)
    a, b = canonical_palette(700, 819)
    np.testing.assert_array_equal(rgb, a)
    np.testing.assert_array_equal(target, b)


def test_rendering_is_reproducible_bounded_and_does_not_relabel_surfaces():
    from chromaseed import canonical_palette, make_palette, render_samples

    rgb, _ = canonical_palette(16, 917)
    clean = render_samples(rgb, 81, "clean")
    rendered = render_samples(rgb, 81, "rendered")
    assert clean.shape == rendered.shape == (16, 128, 3)
    assert np.isfinite(rendered).all() and np.all((rendered >= 0) & (rendered <= 1))
    np.testing.assert_array_equal(rendered, render_samples(rgb, 81, "rendered"))
    assert np.mean(abs(clean - rendered)) > .01
    d = make_palette(64, 731013)
    np.testing.assert_array_equal(d["target"], canonical_palette(64, 731013)[1])
    assert d["clean"].shape == d["rendered"].shape == (64, 36)
    assert np.mean(abs(d["clean"] - d["rendered"])) > .01


def test_compact_payload_roundtrip_matches_torch_forward(tmp_path):
    from chromaseed import X_MEAN, X_STD, Y_MEAN, Y_STD, ChromaSeed, pack_model, predict

    torch.manual_seed(17)
    net = ChromaSeed().eval()
    rng = np.random.default_rng(499)
    x = rng.uniform(.05, .95, (12, 36)).astype(np.float32)
    with torch.no_grad():
        expected = net(torch.from_numpy((x - X_MEAN) / X_STD)).numpy() * Y_STD + Y_MEAN
    model = pack_model(net)
    assert sum(v.size for v in model.values() if v.dtype.kind == "f") == 2749
    np.testing.assert_allclose(predict(model, x), expected, atol=2e-5, rtol=0)
    path = tmp_path / "model.npz"
    np.savez(path, **model)
    with np.load(path, allow_pickle=False) as d:
        loaded = {k: d[k] for k in d.files}
    np.testing.assert_array_equal(predict(loaded, x), predict(model, x))


def test_fixed_normalizers_do_not_depend_on_real_samples():
    from chromaseed import X_MEAN, X_STD, Y_MEAN, Y_STD

    np.testing.assert_array_equal(X_MEAN[:30], np.full(30, .5, dtype=np.float32))
    np.testing.assert_array_equal(X_STD[:30], np.full(30, .25, dtype=np.float32))
    np.testing.assert_array_equal(Y_MEAN, [50, 0, 0])
    np.testing.assert_array_equal(Y_STD, [25, 30, 30])
    assert len(X_MEAN) == len(X_STD) == 36
