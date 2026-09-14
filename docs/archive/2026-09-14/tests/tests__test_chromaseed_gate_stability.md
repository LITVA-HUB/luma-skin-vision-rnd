# `tests/test_chromaseed_gate_stability.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_gate_stability.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Numerical invariants for legal pixel contractions and gate boundaries.

SHA-256 исходника: `dde8166347a41cce6371b6fc6475f57e7d89fcab685b76a9f49ac29361f0f97b`. Строк: **113**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_gate_stability import affine_features, feature_direction, gate_score, projection
from chromaseed_gated_numpy import Predictor
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `stats` | FunctionDef | См. реализацию | [L14](../../../../tests/test_chromaseed_gate_stability.py#L14) |
| `model` | FunctionDef | См. реализацию | [L31](../../../../tests/test_chromaseed_gate_stability.py#L31) |
| `test_affine_matches_real_pixel_statistics` | FunctionDef | См. реализацию | [L48](../../../../tests/test_chromaseed_gate_stability.py#L48) |
| `test_identity_and_bounded_quantiles` | FunctionDef | См. реализацию | [L59](../../../../tests/test_chromaseed_gate_stability.py#L59) |
| `test_bad_transform_rejected` | FunctionDef | См. реализацию | [L68](../../../../tests/test_chromaseed_gate_stability.py#L68) |
| `test_affine_root_predicts_actual_gate_crossing` | FunctionDef | См. реализацию | [L77](../../../../tests/test_chromaseed_gate_stability.py#L77) |
| `test_projection_is_nearest_hyperplane_point` | FunctionDef | См. реализацию | [L88](../../../../tests/test_chromaseed_gate_stability.py#L88) |
| `test_hard_limit_matches_actual_two_sided_prediction` | FunctionDef | См. реализацию | [L103](../../../../tests/test_chromaseed_gate_stability.py#L103) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_affine_matches_real_pixel_statistics` · L48

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('t', [1 / 255, 16 / 255, 64 / 255])
def test_affine_matches_real_pixel_statistics(t):
    rng = np.random.default_rng(917)
    pixels = rng.uniform(0, 1, (151, 3))
    pixels[:, 2] = 0.25
    anchor = np.array([1, 0, 1])
    expected = stats((1 - t) * pixels + t * anchor)
    actual = affine_features(stats(pixels)[None], t, anchor)[0]
    np.testing.assert_allclose(actual, expected, atol=8e-8, rtol=0)
    assert actual[32] == 0 and actual[34] == actual[35] == 0
```

</details>

### `test_identity_and_bounded_quantiles` · L59

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_identity_and_bounded_quantiles():
    x = stats(np.random.default_rng(5).uniform(0, 1, (101, 3))).astype(np.float32)[None]
    np.testing.assert_array_equal(affine_features(x, 0, [0, 0, 0]), x)
    for anchor in ([0, 0, 0], [1, 1, 1], [1, 0, 1]):
        out = affine_features(x, 0.99, anchor)
        assert (out[:, :30] >= 0).all() and (out[:, :30] <= 1).all()
        assert (np.diff(out[:, :27].reshape(-1, 9, 3), axis=1) >= 0).all()
```

</details>

### `test_bad_transform_rejected` · L68

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bad_transform_rejected():
    x = np.zeros((1, 36))
    for t in (-0.01, 1, np.nan):
        with pytest.raises(ValueError):
            affine_features(x, t, [0, 1, 0])
    with pytest.raises(ValueError):
        affine_features(x, 0.1, [0, 2, 0])
```

</details>

### `test_affine_root_predicts_actual_gate_crossing` · L77

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_affine_root_predicts_actual_gate_crossing():
    m = model()
    x = np.full((1, 36), 0.2, dtype=np.float32)
    x[:, 30:33] = 0.1
    slope = (feature_direction(x, [1, 0, 0]) / m["x_std"]) @ m["gate_beta"][1:]
    root = -gate_score(m, x) / slope
    assert root[0] == pytest.approx(0.375, abs=1e-7)
    assert gate_score(m, affine_features(x, float(root[0] - 1e-4), [1, 0, 0]))[0] < 0
    assert gate_score(m, affine_features(x, float(root[0] + 1e-4), [1, 0, 0]))[0] > 0
```

</details>

### `test_projection_is_nearest_hyperplane_point` · L88

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_projection_is_nearest_hyperplane_point():
    m = model()
    m["gate_beta"][2] = 2
    x = np.full((2, 36), 0.2, np.float32)
    z, score, projected, rms = projection(m, x)
    beta = m["gate_beta"].astype(np.float64)
    np.testing.assert_allclose(projected @ beta[1:] + beta[0], 0, atol=1e-15)
    np.testing.assert_allclose(
        np.linalg.norm(projected - z, axis=1), np.abs(score) / np.linalg.norm(beta[1:]), atol=1e-15
    )
    np.testing.assert_allclose(rms, np.linalg.norm(projected - z, axis=1) / 6, atol=1e-15)
    tangent = np.r_[2, -1, np.zeros(34)]
    assert np.all(np.linalg.norm(projected + tangent - z, axis=1) > rms * 6)
```

</details>

### `test_hard_limit_matches_actual_two_sided_prediction` · L103

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_hard_limit_matches_actual_two_sided_prediction():
    m = model()
    p = Predictor(m)
    x = np.zeros(36, np.float32)
    x[0] = 0.5
    left, right = x.copy(), x.copy()
    left[0] -= 1e-5
    right[0] += 1e-5
    observed = p(right) - p(left)
    expected = 2 * float(m["rho"]) * np.exp(-0.5 * 0.25 / 36) * m["correction"][0]
    np.testing.assert_allclose(observed, expected, atol=2e-7, rtol=0)
```

</details>
