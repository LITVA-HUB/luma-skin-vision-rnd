# `tests/unit/test_measurement.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/unit/test_measurement.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `d12b8e1590bca0a6ab0d7d6d33e8abd713bcb7e27eabffbb1301345075b10255`. Строк: **44**.

## Зависимости

```python
import numpy as np
import pytest
from luma_skin_vision.photometry import apply_ccm, fit_ccm, hypotheses
from luma_skin_vision.roi import ambiguity_features, cheek_masks, measure
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_cheeks_and_uniform_measurement` | FunctionDef | См. реализацию | [L8](../../../../tests/unit/test_measurement.py#L8) |
| `test_bounded_hypotheses_and_feature_finiteness` | FunctionDef | См. реализацию | [L21](../../../../tests/unit/test_measurement.py#L21) |
| `test_train_only_ccm` | FunctionDef | См. реализацию | [L32](../../../../tests/unit/test_measurement.py#L32) |
| `test_anatomical_left_is_image_right_unmirrored` | FunctionDef | См. реализацию | [L41](../../../../tests/unit/test_measurement.py#L41) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_cheeks_and_uniform_measurement` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_cheeks_and_uniform_measurement():
    masks = cheek_masks((100, 100), [10, 5, 80, 90])
    assert len(masks) == 2 and all(m.sum() > 50 for m in masks)
    assert not np.any(masks[0] & masks[1])
    rgb = np.full((100, 100, 3), [0.7, 0.5, 0.4])
    # Independent OpenCV float conversion has LUT quantization (~0.2 Lab units).
    np.testing.assert_allclose(measure(rgb, masks), [57.818604, 16.375, 21.859375], atol=0.2)
    with pytest.raises(ValueError):
        cheek_masks((100, 100), [-1, 0, 10, 10])
    with pytest.raises(ValueError, match="usable"):
        measure(np.zeros_like(rgb), masks)
```

</details>

### `test_bounded_hypotheses_and_feature_finiteness` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bounded_hypotheses_and_feature_finiteness():
    rgb = np.random.default_rng(1).uniform(0.15, 0.8, (64, 64, 3))
    hs = hypotheses(rgb)
    np.testing.assert_array_equal(hs["none"], rgb)
    assert set(hs) == {"none", "gray_world", "shades_of_gray"}
    assert all(np.isfinite(h).all() and h.min() >= 0 and h.max() <= 1 for h in hs.values())
    masks = cheek_masks((64, 64), [0, 0, 64, 64])
    features = ambiguity_features(rgb, masks)
    assert features.shape == (12,) and np.isfinite(features).all()
```

</details>

### `test_train_only_ccm` · L32

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_train_only_ccm():
    x = np.random.default_rng(2).uniform(0.1, 0.7, (40, 3))
    y = x @ np.diag([0.95, 1.02, 1.04]) + 0.01
    ccm = fit_ccm(x, y, split="train", ridge=1e-8)
    np.testing.assert_allclose(apply_ccm(x, ccm), y, atol=1e-6)
    with pytest.raises(ValueError, match="train"):
        fit_ccm(x, y, split="test")
```

</details>

### `test_anatomical_left_is_image_right_unmirrored` · L41

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_anatomical_left_is_image_right_unmirrored():
    left, right = cheek_masks((100, 100), [0, 0, 100, 100])
    assert np.where(left)[1].mean() > 50
    assert np.where(right)[1].mean() < 50
```

</details>
