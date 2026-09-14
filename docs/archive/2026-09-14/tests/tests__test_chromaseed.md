# `tests/test_chromaseed.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent numerical and provenance checks for synthetic palette transfer.

SHA-256 исходника: `797cf911d796da65d3cec3575076c19f27522f77da8b8713a4438b9a44af52e8`. Строк: **87**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_color36_matches_independent_quantiles_and_correlations` | FunctionDef | См. реализацию | [L11](../../../../tests/test_chromaseed.py#L11) |
| `test_canonical_palette_has_exact_colorimetric_targets_and_gamut` | FunctionDef | См. реализацию | [L31](../../../../tests/test_chromaseed.py#L31) |
| `test_rendering_is_reproducible_bounded_and_does_not_relabel_surfaces` | FunctionDef | См. реализацию | [L45](../../../../tests/test_chromaseed.py#L45) |
| `test_compact_payload_roundtrip_matches_torch_forward` | FunctionDef | См. реализацию | [L61](../../../../tests/test_chromaseed.py#L61) |
| `test_fixed_normalizers_do_not_depend_on_real_samples` | FunctionDef | См. реализацию | [L80](../../../../tests/test_chromaseed.py#L80) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_color36_matches_independent_quantiles_and_correlations` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_canonical_palette_has_exact_colorimetric_targets_and_gamut` · L31

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_rendering_is_reproducible_bounded_and_does_not_relabel_surfaces` · L45

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_compact_payload_roundtrip_matches_torch_forward` · L61

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
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
```

</details>

### `test_fixed_normalizers_do_not_depend_on_real_samples` · L80

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fixed_normalizers_do_not_depend_on_real_samples():
    from chromaseed import X_MEAN, X_STD, Y_MEAN, Y_STD

    np.testing.assert_array_equal(X_MEAN[:30], np.full(30, .5, dtype=np.float32))
    np.testing.assert_array_equal(X_STD[:30], np.full(30, .25, dtype=np.float32))
    np.testing.assert_array_equal(Y_MEAN, [50, 0, 0])
    np.testing.assert_array_equal(Y_STD, [25, 30, 30])
    assert len(X_MEAN) == len(X_STD) == 36
```

</details>
