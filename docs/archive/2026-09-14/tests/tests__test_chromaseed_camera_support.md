# `tests/test_chromaseed_camera_support.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_camera_support.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `4a77b110dd523ad73d85de3e56e9ce2e51b7f7624c7422345b399d7dde4e8c76`. Строк: **126**.

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
| `test_camera_person_site_weights_and_person_reduction` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_camera_support.py#L10) |
| `test_fit_standardizer_does_not_see_query_extremes` | FunctionDef | См. реализацию | [L23](../../../../tests/test_chromaseed_camera_support.py#L23) |
| `test_linear_and_kernel_scores_match_independent_lstsq` | FunctionDef | См. реализацию | [L35](../../../../tests/test_chromaseed_camera_support.py#L35) |
| `test_linear_target_residual_is_orthogonal_on_fit_only` | FunctionDef | См. реализацию | [L67](../../../../tests/test_chromaseed_camera_support.py#L67) |
| `test_auc_ties_and_person_counts` | FunctionDef | См. реализацию | [L82](../../../../tests/test_chromaseed_camera_support.py#L82) |
| `test_assignment_maximizes_cardinality_before_distance` | FunctionDef | См. реализацию | [L91](../../../../tests/test_chromaseed_camera_support.py#L91) |
| `test_support_excludes_every_row_of_same_person` | FunctionDef | См. реализацию | [L101](../../../../tests/test_chromaseed_camera_support.py#L101) |
| `test_weighted_empirical_quantile_uses_cumulative_mass` | FunctionDef | См. реализацию | [L110](../../../../tests/test_chromaseed_camera_support.py#L110) |
| `test_invalid_normalization_weights_fail` | FunctionDef | См. реализацию | [L122](../../../../tests/test_chromaseed_camera_support.py#L122) |

## Все тестовые определения (9)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_camera_person_site_weights_and_person_reduction` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_camera_person_site_weights_and_person_reduction():
    from chromaseed_camera_support import aggregate_people, row_weights

    p = np.array([0, 0, 0, 1, 2, 2])
    s = np.array([0, 0, 1, 0, 0, 0])
    c = np.array([0, 0, 0, 0, 1, 1])
    w = row_weights(p, s, c)
    np.testing.assert_allclose(w, [0.0625, 0.0625, 0.125, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(
        aggregate_people(np.array([0.0, 2.0, 5.0, 7.0, 0.0, 4.0]), p, s), [3.0, 7.0, 2.0]
    )
```

</details>

### `test_fit_standardizer_does_not_see_query_extremes` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fit_standardizer_does_not_see_query_extremes():
    from chromaseed_camera_support import standardize

    x = np.array([[0.0, 1.0], [2.0, 1.0]])
    q = np.array([[1e9, 4.0]])
    z, v, prep = standardize(x, q, np.array([0.5, 0.5]))
    np.testing.assert_array_equal(z, [[-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(prep["mean"], [1.0, 1.0])
    np.testing.assert_allclose(prep["std"], [1.0, 1e-6])
    assert v[0, 0] == 1e9 - 1
```

</details>

### `test_linear_and_kernel_scores_match_independent_lstsq` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_linear_and_kernel_scores_match_independent_lstsq():
    from chromaseed_camera_support import classify, kernel

    rng = np.random.default_rng(217)
    x = rng.normal(size=(21, 5))
    q = rng.normal(size=(7, 5))
    y = rng.choice([-1.0, 1.0], 21)
    w = rng.uniform(0.2, 1, 21)
    w /= w.sum()
    actual, info = classify(x, q, y, w, "linear")
    design = np.column_stack([np.ones(21), x])
    penalty = np.column_stack([np.zeros(5), np.sqrt(0.1) * np.eye(5)])
    coefficient = np.linalg.lstsq(
        np.vstack([np.sqrt(w)[:, None] * design, penalty]),
        np.r_[np.sqrt(w) * y, np.zeros(5)],
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        actual, np.column_stack([np.ones(7), q]) @ coefficient, atol=1e-11, rtol=0
    )
    actual, info = classify(x, q, y, w, "rbf")
    k = kernel(x, x, info["width"])
    root = np.sqrt(w)
    coeff = (
        np.linalg.lstsq(
            root[:, None] * k * root[None, :] + 0.01 * np.eye(21), root * y, rcond=None
        )[0]
        * root
    )
    np.testing.assert_allclose(actual, kernel(q, x, info["width"]) @ coeff, atol=1e-11, rtol=0)
```

</details>

### `test_linear_target_residual_is_orthogonal_on_fit_only` · L67

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_linear_target_residual_is_orthogonal_on_fit_only():
    from chromaseed_camera_support import views

    rng = np.random.default_rng(307)
    y = rng.normal(size=(25, 3))
    x = y @ rng.normal(size=(3, 36)) + rng.normal(size=(25, 36)) * 0.02
    w = np.ones(25) / 25
    fit, query = views(x, y, x[:2], y[:2] + 100.0, w)
    assert list(fit) == ["color36", "rgb_mean", "lab3", "lab_residual"]
    np.testing.assert_allclose(
        fit["lab3"].T @ (w[:, None] * fit["lab_residual"]), 0.0, atol=1e-10, rtol=0
    )
    assert np.abs(query["lab_residual"]).max() > 100.0
```

</details>

### `test_auc_ties_and_person_counts` · L82

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_auc_ties_and_person_counts():
    from chromaseed_camera_support import classification_metrics

    m = classification_metrics(np.array([0.0, 1.0, 0.0, 0.0]), np.array([1, 1, -1, -1]))
    assert m["auc"] == 0.75 and m["balanced_accuracy"] == 0.5
    assert m["slr_correct"] == 2 and m["ipod_correct"] == 0
    assert classification_metrics(np.array([]), np.array([])) is None
```

</details>

### `test_assignment_maximizes_cardinality_before_distance` · L91

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_assignment_maximizes_cardinality_before_distance():
    from chromaseed_camera_support import match_cost

    row, col = match_cost(np.array([[1.0, 2.0], [1.1, 100.0]]), 2.0)
    np.testing.assert_array_equal(row, [0, 1])
    np.testing.assert_array_equal(col, [1, 0])
    row, col = match_cost(np.array([[1.0, 2.0], [1.1, 100.0]]), 0.5)
    assert len(row) == len(col) == 0
```

</details>

### `test_support_excludes_every_row_of_same_person` · L101

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_support_excludes_every_row_of_same_person():
    from chromaseed_camera_support import nearest

    x = np.array([[0.0], [0.0], [2.0], [2.0]])
    person = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(nearest(x, x, "rms", person, person), [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(nearest(np.array([[1.0]]), x, "rms"), [1.0])
```

</details>

### `test_weighted_empirical_quantile_uses_cumulative_mass` · L110

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_weighted_empirical_quantile_uses_cumulative_mass():
    from chromaseed_camera_support import weighted_quantile

    x = np.array([100.0, 0.0, 10.0])
    w = np.array([0.02, 0.90, 0.08])
    assert weighted_quantile(x, w, 0.95) == 10.0
    assert weighted_quantile(x, w, 0.5) == 0.0
```

</details>

### `test_invalid_normalization_weights_fail` · L122

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('bad', [np.array([np.nan, 1.0]), np.array([-1.0, 2.0]), np.array([0.0, 0.0])])
def test_invalid_normalization_weights_fail(bad):
    from chromaseed_camera_support import standardize

    with pytest.raises(ValueError):
        standardize(np.ones((2, 3)), np.ones((1, 3)), bad)
```

</details>
