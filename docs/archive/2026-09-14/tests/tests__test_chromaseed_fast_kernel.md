# `tests/test_chromaseed_fast_kernel.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_fast_kernel.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent numeric behavior checks for on-demand compact training.

SHA-256 исходника: `e1e209a58f4edf088994d8e60019ffac9110ba67a34a9bc92b91ea6b2f6465b3`. Строк: **114**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_fast_kernel import (
    KernelColumns,
    fit_one,
    pair_indices,
    sampled_width,
    solve_columns,
    streaming_landmarks,
)
from chromaseed_kernel import (
    gaussian_kernel,
    nystrom_coefficients,
    predict_kernel,
    select_landmarks,
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_pair_stream_is_prefix_without_self_pairs` | FunctionDef | См. реализацию | [L25](../../../../tests/test_chromaseed_fast_kernel.py#L25) |
| `test_sampled_width_matches_direct_pair_distances` | FunctionDef | См. реализацию | [L34](../../../../tests/test_chromaseed_fast_kernel.py#L34) |
| `test_width_degenerate_and_linear_fallback` | FunctionDef | См. реализацию | [L45](../../../../tests/test_chromaseed_fast_kernel.py#L45) |
| `test_streaming_cholesky_matches_dense_nonuniform_weights` | FunctionDef | См. реализацию | [L57](../../../../tests/test_chromaseed_fast_kernel.py#L57) |
| `test_streaming_nystrom_matches_dense_weighted_readout` | FunctionDef | См. реализацию | [L73](../../../../tests/test_chromaseed_fast_kernel.py#L73) |
| `test_column_requests_stop_at_available_rank_and_reject_bad_weights` | FunctionDef | См. реализацию | [L84](../../../../tests/test_chromaseed_fast_kernel.py#L84) |
| `test_end_to_end_exact_control_preserves_unseen_predictions` | FunctionDef | См. реализацию | [L95](../../../../tests/test_chromaseed_fast_kernel.py#L95) |
| `test_sampled_fit_reports_linear_matrices_and_real_pair_budget` | FunctionDef | См. реализацию | [L107](../../../../tests/test_chromaseed_fast_kernel.py#L107) |

## Все тестовые определения (8)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_pair_stream_is_prefix_without_self_pairs` · L25

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pair_stream_is_prefix_without_self_pairs():
    small = pair_indices(137, 1024, 17)
    large = pair_indices(137, 16384, 17)
    np.testing.assert_array_equal(small, large[:len(small)])
    assert small.shape == (1024, 2)
    assert np.all(small[:, 0] != small[:, 1])
    assert small.min() >= 0 and small.max() < 137
```

</details>

### `test_sampled_width_matches_direct_pair_distances` · L34

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_sampled_width_matches_direct_pair_distances():
    x = np.random.default_rng(4).normal(size=(63, 36))
    pairs = pair_indices(len(x), 1024, 104746)
    positive = np.sqrt(np.mean((x[pairs[:, 0]] - x[pairs[:, 1]]) ** 2, axis=1))
    expected = np.sort(positive[positive > 1e-10])[(len(positive) - 1) // 2]
    width, info = sampled_width(x, 1024, 17)
    assert width == pytest.approx(expected, abs=1e-14)
    assert info["fallback"] == "none"
    assert info["requested_pairs"] == 1024
```

</details>

### `test_width_degenerate_and_linear_fallback` · L45

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_width_degenerate_and_linear_fallback():
    assert sampled_width(np.ones((24, 36)), 1024, 17)[0] == 1e-6
    x = np.zeros((300, 36))
    pairs = pair_indices(len(x), 1, 104746)
    rare = next(i for i in range(1, len(x)) if i not in pairs)
    x[rare] = 2.
    width, info = sampled_width(x, 1, 17)
    assert info["fallback"] == "anchor"
    assert width == 2.
```

</details>

### `test_streaming_cholesky_matches_dense_nonuniform_weights` · L57

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('seed', [17, 29, 43])
def test_streaming_cholesky_matches_dense_nonuniform_weights(seed):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(131, 36))
    weights = rng.uniform(.2, 2, size=len(x))
    kernel = gaussian_kernel(x, x, .8)
    expected, factor, remainder = select_landmarks(kernel, weights, "rpchol", 32, seed)
    operator = KernelColumns(x, .8)
    selected, columns, actual_factor, residual, info = streaming_landmarks(operator, weights, 32, seed)
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_allclose(columns, kernel[:, expected], atol=2e-14, rtol=0)
    np.testing.assert_allclose(actual_factor @ actual_factor.T, factor @ factor.T, atol=1e-12)
    np.testing.assert_allclose(residual, remainder, atol=1e-12)
    assert operator.entries_evaluated == len(x) * len(selected)
    assert info["largest_training_matrix_shape"] == [len(x), 32]
```

</details>

### `test_streaming_nystrom_matches_dense_weighted_readout` · L73

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_streaming_nystrom_matches_dense_weighted_readout():
    rng = np.random.default_rng(28)
    x, y = rng.normal(size=(109, 36)), rng.normal(size=(109, 3))
    weights = rng.uniform(.1, 3., size=len(x))
    operator = KernelColumns(x, .73)
    ids, columns, _, _, _ = streaming_landmarks(operator, weights, 48, 17)
    actual, _ = solve_columns(columns, ids, y, weights, (.1, 1., 10.))
    expected, _ = nystrom_coefficients(gaussian_kernel(x, x, .73), ids, y, weights, (.1, 1., 10.))
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)
```

</details>

### `test_column_requests_stop_at_available_rank_and_reject_bad_weights` · L84

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_column_requests_stop_at_available_rank_and_reject_bad_weights():
    x = np.ones((15, 36))
    operator = KernelColumns(x, 1.)
    ids, columns, _, residual, _ = streaming_landmarks(operator, np.ones(len(x)), 32, 17)
    assert len(ids) == 1 and columns.shape == (15, 1)
    assert operator.entries_evaluated == 15
    assert np.max(residual) == 0
    with pytest.raises(ValueError):
        streaming_landmarks(operator, np.zeros(len(x)), 8, 17)
```

</details>

### `test_end_to_end_exact_control_preserves_unseen_predictions` · L95

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_end_to_end_exact_control_preserves_unseen_predictions():
    rng = np.random.default_rng(811)
    x = rng.uniform(0, 1, size=(134, 36)).astype(np.float32)
    y = rng.normal([50, 5, 10], [12, 6, 9], size=(len(x), 3))
    w = rng.uniform(.2, 2, size=len(x))
    dense, _ = fit_one(x, y, w, "dense_exact", 64, 17, 1, 1)
    streamed, _ = fit_one(x, y, w, "column_exact", 64, 17, 1, 1)
    query = rng.uniform(0, 1, size=(23, 36)).astype(np.float32)
    np.testing.assert_array_equal(dense["centers"], streamed["centers"])
    np.testing.assert_allclose(predict_kernel(dense, query), predict_kernel(streamed, query), atol=2e-6, rtol=0)
```

</details>

### `test_sampled_fit_reports_linear_matrices_and_real_pair_budget` · L107

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_sampled_fit_reports_linear_matrices_and_real_pair_budget():
    rng = np.random.default_rng(50)
    x, y = rng.normal(size=(129, 36)), rng.normal(size=(129, 3))
    model, info = fit_one(x, y, np.ones(len(x)), "column_pairs1024", 32, 17, 1, 1)
    assert info["width_info"]["requested_pairs"] == 1024
    assert info["kernel_entries_evaluated"] == 129 * 32
    assert info["largest_training_matrix_shape"] == [129, 32]
    assert model["centers"].shape == (32, 36)
```

</details>
