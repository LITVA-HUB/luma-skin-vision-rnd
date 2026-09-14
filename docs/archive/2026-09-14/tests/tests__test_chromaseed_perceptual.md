# `tests/test_chromaseed_perceptual.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_perceptual.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `bb712ba0f8ccb98040673f700b4490aec27cd14050d9d415324364acd448e7e6`. Строк: **84**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_published_color_pairs` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_perceptual.py#L10) |
| `test_tensor_predicts_small_direction_color_differences` | FunctionDef | См. реализацию | [L17](../../../../tests/test_chromaseed_perceptual.py#L17) |
| `test_neutral_tensor_is_finite_and_expected` | FunctionDef | См. реализацию | [L31](../../../../tests/test_chromaseed_perceptual.py#L31) |
| `test_tensor_matches_independent_analytic_geometry` | FunctionDef | См. реализацию | [L37](../../../../tests/test_chromaseed_perceptual.py#L37) |
| `test_coupled_ridge_matches_independent_augmented_lstsq` | FunctionDef | См. реализацию | [L45](../../../../tests/test_chromaseed_perceptual.py#L45) |
| `test_refinement_is_monotone_and_zero_is_baseline` | FunctionDef | См. реализацию | [L69](../../../../tests/test_chromaseed_perceptual.py#L69) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_published_color_pairs` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_published_color_pairs():
    from luma_skin_vision.color import delta_e00
    data = np.loadtxt(Path(__file__).parent / "fixtures/ciede2000_sharma.txt")
    assert len(data) == 34
    np.testing.assert_allclose(delta_e00(data[:, :3], data[:, 3:6]), data[:, 6], rtol=0, atol=5e-5)
```

</details>

### `test_tensor_predicts_small_direction_color_differences` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_tensor_predicts_small_direction_color_differences():
    from chromaseed_perceptual import local_tensor

    from luma_skin_vision.color import delta_e00
    rng = np.random.default_rng(38)
    anchors = rng.uniform([15, -45, -60], [85, 50, 70], (90, 3))
    vectors = rng.normal(size=anchors.shape) * 1e-4
    metric, info = local_tensor(anchors)
    predicted = np.einsum("ni,nij,nj->n", vectors, metric, vectors)
    np.testing.assert_allclose(predicted, delta_e00(anchors, anchors + vectors) ** 2, rtol=4e-5, atol=1e-12)
    assert info["clipped_eigenvalues"] == 0
    np.testing.assert_allclose(metric, local_tensor(anchors, h=2e-4)[0], rtol=2e-5, atol=1e-8)
```

</details>

### `test_neutral_tensor_is_finite_and_expected` · L31

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_neutral_tensor_is_finite_and_expected():
    from chromaseed_perceptual import local_tensor
    metric, _ = local_tensor(np.array([[50., 0., 0.]]))
    np.testing.assert_allclose(metric[0], np.diag([1., 2.25, 1.]), atol=2e-4)
```

</details>

### `test_tensor_matches_independent_analytic_geometry` · L37

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_tensor_matches_independent_analytic_geometry():
    from chromaseed_perceptual import local_tensor
    from chromaseed_perceptual_reference import analytic_tensor
    rng = np.random.default_rng(615)
    lab = rng.uniform([10, -80, -80], [95, 80, 80], size=(131, 3))
    np.testing.assert_allclose(local_tensor(lab)[0], analytic_tensor(lab), rtol=3e-5, atol=1e-7)
```

</details>

### `test_coupled_ridge_matches_independent_augmented_lstsq` · L45

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_coupled_ridge_matches_independent_augmented_lstsq():
    from chromaseed_perceptual import coupled_ridge
    rng = np.random.default_rng(70)
    z, y = rng.normal(size=(30, 7)), rng.normal(size=(30, 3))
    a = rng.normal(size=(30, 3, 3))
    g = a @ a.transpose(0, 2, 1) + np.eye(3)
    w = rng.uniform(.1, 2, 30)
    alpha = .7
    answer, residual = coupled_ridge(z, y, g, w, alpha)
    rows, targets = [], []
    for zi, yi, gi, wi in zip(z, y, g, w, strict=True):
        left = np.linalg.cholesky(gi).T * np.sqrt(wi)
        rows.append(left @ np.kron(zi[None], np.eye(3)))
        targets.append(left @ yi)
    design = np.vstack([*rows, np.sqrt(alpha) * np.eye(21)])
    target = np.concatenate([*targets, np.zeros(21)])
    expected = np.linalg.lstsq(design, target, rcond=None)[0].reshape(7, 3)
    np.testing.assert_allclose(answer, expected, rtol=1e-10, atol=1e-10)
    from chromaseed_perceptual_reference import augmented_svd
    np.testing.assert_allclose(augmented_svd(z, y, g, w, alpha), expected, rtol=1e-10, atol=1e-10)
    assert residual < 1e-10
```

</details>

### `test_refinement_is_monotone_and_zero_is_baseline` · L69

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family', ['local_irls', 'midpoint_irls'])
def test_refinement_is_monotone_and_zero_is_baseline(family):
    from chromaseed_condensed_exact import fit_condensed
    from chromaseed_perceptual import fit_single
    rng = np.random.default_rng(725)
    x = rng.normal(size=(53, 36)).astype(np.float32)
    y = rng.normal([50, 10, 17], [8, 3, 7], size=(53, 3))
    w = rng.uniform(.5, 2, 53)
    zero, _ = fit_single(x, y, w, family, 17, 1, 0, 0, rank=18)
    baseline, _ = fit_condensed(x, y, w, 18, 17, 1, 0)
    for field in baseline:
        np.testing.assert_array_equal(zero[field], baseline[field])
    model, info = fit_single(x, y, w, family, 17, 1, 0, 4, rank=18)
    assert sum(v.nbytes for v in model.values()) == sum(v.nbytes for v in baseline.values())
    values = [item["objective"] for item in info["trajectory"]]
    assert len(values) == 5 and np.all(np.diff(values) <= 1e-9)
    assert info["executed_solves"] <= 4
```

</details>
