# `tests/test_skin_local_search_core.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_local_search_core.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent numerical checks for the local closed-form search core.

SHA-256 исходника: `9cbd9a3b537c813bf8756e8f919017adffd70be1aaabf6afd0137d219ac1af93`. Строк: **220**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_numpy_ridge` | FunctionDef | См. реализацию | [L13](../../../../tests/test_skin_local_search_core.py#L13) |
| `_numpy_objective` | FunctionDef | См. реализацию | [L22](../../../../tests/test_skin_local_search_core.py#L22) |
| `test_weighted_ridge_and_objective_match_independent_numpy_solution` | FunctionDef | См. реализацию | [L31](../../../../tests/test_skin_local_search_core.py#L31) |
| `test_rbf_features_match_numpy_and_support_per_center_widths` | FunctionDef | См. реализацию | [L61](../../../../tests/test_skin_local_search_core.py#L61) |
| `_exhaustive_greedy` | FunctionDef | См. реализацию | [L73](../../../../tests/test_skin_local_search_core.py#L73) |
| `test_greedy_schur_gains_match_exhaustive_ridge_refits_at_every_step` | FunctionDef | См. реализацию | [L98](../../../../tests/test_skin_local_search_core.py#L98) |
| `test_greedy_objective_is_monotone_and_duplicate_atoms_remain_stable` | FunctionDef | См. реализацию | [L123](../../../../tests/test_skin_local_search_core.py#L123) |
| `test_weighted_kernel_ridge_matches_numpy_and_predicts_with_same_kernel` | FunctionDef | См. реализацию | [L150](../../../../tests/test_skin_local_search_core.py#L150) |
| `test_public_functions_require_float64_and_positive_regularization` | FunctionDef | См. реализацию | [L172](../../../../tests/test_skin_local_search_core.py#L172) |
| `test_rbf_cpu_gpu_float64_parity` | FunctionDef | См. реализацию | [L211](../../../../tests/test_skin_local_search_core.py#L211) |

## Все тестовые определения (7)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_weighted_ridge_and_objective_match_independent_numpy_solution` · L31

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_weighted_ridge_and_objective_match_independent_numpy_solution():
    from skin_local_search_core import regularized_objective, ridge_solve

    rng = np.random.default_rng(2201)
    design = np.column_stack([np.ones(19), rng.normal(size=(19, 4))])
    target = rng.normal(size=(19, 3))
    weights = rng.uniform(0.2, 2.0, size=19)
    expected = _numpy_ridge(design, target, 0.7, weights)

    beta = ridge_solve(torch.tensor(design), torch.tensor(target), 0.7, torch.tensor(weights))
    np.testing.assert_allclose(beta.numpy(), expected, rtol=2e-12, atol=2e-12)
    got = regularized_objective(
        torch.tensor(design), torch.tensor(target), beta, 0.7, torch.tensor(weights)
    )
    assert got.dtype == torch.float64
    assert float(got) == pytest.approx(
        _numpy_objective(design, target, expected, 0.7, weights), rel=2e-12, abs=2e-12
    )

    shifted = target + np.array([13.0, -4.0, 2.5])
    shifted_beta = ridge_solve(torch.tensor(design), torch.tensor(shifted), 0.7)
    np.testing.assert_allclose(
        shifted_beta[0].numpy(),
        ridge_solve(torch.tensor(design), torch.tensor(target), 0.7)[0].numpy()
        + np.array([13.0, -4.0, 2.5]),
        rtol=2e-12,
        atol=2e-12,
    )
```

</details>

### `test_rbf_features_match_numpy_and_support_per_center_widths` · L61

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_rbf_features_match_numpy_and_support_per_center_widths():
    from skin_local_search_core import rbf_features

    x = np.array([[0.0, 1.0], [2.0, -1.0], [0.5, 0.25]])
    centers = np.array([[0.0, 0.0], [1.0, -2.0]])
    widths = np.array([0.5, 1.75])
    expected = np.exp(-0.5 * np.mean((x[:, None] - centers[None]) ** 2, axis=2) / widths**2)
    got = rbf_features(torch.tensor(x), torch.tensor(centers), torch.tensor(widths))
    assert got.dtype == torch.float64
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-14, atol=1e-14)
```

</details>

### `test_greedy_schur_gains_match_exhaustive_ridge_refits_at_every_step` · L98

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('weighted', [False, True])
def test_greedy_schur_gains_match_exhaustive_ridge_refits_at_every_step(weighted):
    from skin_local_search_core import greedy_ridge_indices

    rng = np.random.default_rng(902 if weighted else 901)
    base = np.column_stack([np.ones(23), rng.normal(size=(23, 2))])
    atoms = rng.normal(size=(23, 6))
    target = rng.normal(size=(23, 3))
    weights = rng.uniform(0.1, 3.0, size=23) if weighted else None
    expected_indices, expected_gains = _exhaustive_greedy(
        base, atoms, target, 0.35, 6, weights
    )
    got_indices, got_gains = greedy_ridge_indices(
        torch.tensor(base),
        torch.tensor(atoms),
        torch.tensor(target),
        0.35,
        6,
        None if weights is None else torch.tensor(weights),
    )
    assert got_indices == expected_indices
    np.testing.assert_allclose(got_gains, expected_gains, rtol=2e-10, atol=2e-11)
    assert len(got_indices) == len(set(got_indices))
    assert np.isfinite(got_gains).all()
```

</details>

### `test_greedy_objective_is_monotone_and_duplicate_atoms_remain_stable` · L123

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_greedy_objective_is_monotone_and_duplicate_atoms_remain_stable():
    from skin_local_search_core import greedy_ridge_indices, regularized_objective, ridge_solve

    rng = np.random.default_rng(881)
    base = np.column_stack([np.ones(15), rng.normal(size=(15, 1))])
    first = rng.normal(size=15)
    atoms = np.column_stack([first, first, first + 1e-13, np.ones(15), rng.normal(size=15)])
    target = rng.normal(size=(15, 2))
    indices, gains = greedy_ridge_indices(
        torch.tensor(base), torch.tensor(atoms), torch.tensor(target), 1e-4, 20
    )
    assert len(indices) == atoms.shape[1]
    assert len(indices) == len(set(indices))
    assert np.isfinite(gains).all()
    assert min(gains) >= -1e-9

    objectives = []
    for count in range(len(indices) + 1):
        design = np.column_stack([base, atoms[:, indices[:count]]])
        beta = ridge_solve(torch.tensor(design), torch.tensor(target), 1e-4)
        objectives.append(float(regularized_objective(
            torch.tensor(design), torch.tensor(target), beta, 1e-4
        )))
    assert np.all(np.diff(objectives) <= 1e-9)
    np.testing.assert_allclose(np.array(objectives[:-1]) - objectives[1:], gains, atol=2e-9)
```

</details>

### `test_weighted_kernel_ridge_matches_numpy_and_predicts_with_same_kernel` · L150

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_weighted_kernel_ridge_matches_numpy_and_predicts_with_same_kernel():
    from skin_local_search_core import kernel_ridge_fit, kernel_ridge_predict

    rng = np.random.default_rng(771)
    x = rng.normal(size=(11, 3))
    query = rng.normal(size=(5, 3))
    target = rng.normal(size=(11, 2))
    weights = rng.uniform(0.25, 2.5, size=11)
    width = 1.3
    alpha = 0.08
    kernel = np.exp(-0.5 * np.mean((x[:, None] - x[None]) ** 2, axis=2) / width**2)
    expected_coeff = np.linalg.solve(kernel + alpha * np.diag(1.0 / weights), target)
    cross = np.exp(-0.5 * np.mean((query[:, None] - x[None]) ** 2, axis=2) / width**2)

    coeff = kernel_ridge_fit(
        torch.tensor(x), torch.tensor(target), width, alpha, torch.tensor(weights)
    )
    prediction = kernel_ridge_predict(torch.tensor(query), torch.tensor(x), coeff, width)
    np.testing.assert_allclose(coeff.numpy(), expected_coeff, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(prediction.numpy(), cross @ expected_coeff, rtol=2e-11, atol=2e-11)
```

</details>

### `test_public_functions_require_float64_and_positive_regularization` · L172

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_public_functions_require_float64_and_positive_regularization():
    from skin_local_search_core import (
        greedy_ridge_indices,
        kernel_ridge_fit,
        rbf_features,
        ridge_solve,
    )

    x32 = torch.ones((4, 2), dtype=torch.float32)
    y64 = torch.ones((4, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="float64"):
        ridge_solve(x32, y64, 1.0)
    with pytest.raises(ValueError, match="alpha"):
        ridge_solve(torch.ones((4, 2), dtype=torch.float64), y64, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        greedy_ridge_indices(
            torch.ones((4, 2), dtype=torch.float64),
            torch.ones((4, 2), dtype=torch.float64),
            y64,
            -1.0,
            1,
        )
    with pytest.raises(ValueError, match="width"):
        rbf_features(
            torch.ones((4, 2), dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.float64),
            0.0,
        )
    with pytest.raises(ValueError, match="weight"):
        kernel_ridge_fit(
            torch.ones((4, 2), dtype=torch.float64),
            y64,
            1.0,
            0.1,
            torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64),
        )
```

</details>

### `test_rbf_cpu_gpu_float64_parity` · L211

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_rbf_cpu_gpu_float64_parity():
    from skin_local_search_core import rbf_features

    generator = torch.Generator().manual_seed(91)
    x = torch.randn((17, 5), generator=generator, dtype=torch.float64)
    centers = torch.randn((7, 5), generator=generator, dtype=torch.float64)
    widths = torch.linspace(0.4, 1.7, 7, dtype=torch.float64)
    cpu = rbf_features(x, centers, widths)
    gpu = rbf_features(x.cuda(), centers.cuda(), widths.cuda()).cpu()
    torch.testing.assert_close(gpu, cpu, rtol=2e-13, atol=2e-13)
```

</details>
