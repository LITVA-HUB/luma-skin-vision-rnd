# `tests/test_chromaseed_kernel.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_kernel.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `9bad457558eab7c1732695bcc929262376ba084c25093c5e683ff37f32373fbc`. Строк: **162**.

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
| `test_nystrom_all_landmarks_matches_independent_weighted_kernel_ridge` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_kernel.py#L10) |
| `test_nystrom_matches_direct_rkhs_regularized_normal_equations` | FunctionDef | См. реализацию | [L23](../../../../tests/test_chromaseed_kernel.py#L23) |
| `test_median_width_excludes_exact_duplicates_without_distance_cancellation` | FunctionDef | См. реализацию | [L36](../../../../tests/test_chromaseed_kernel.py#L36) |
| `test_landmark_prefixes_reconstruct_psd_residual_and_replay` | FunctionDef | См. реализацию | [L44](../../../../tests/test_chromaseed_kernel.py#L44) |
| `test_projection_diagnostic_bounds_arbitrary_rounded_residuals` | FunctionDef | См. реализацию | [L61](../../../../tests/test_chromaseed_kernel.py#L61) |
| `test_adaptive_kernel_computes_only_new_centers_and_honors_forced_exit` | FunctionDef | См. реализацию | [L89](../../../../tests/test_chromaseed_kernel.py#L89) |
| `test_convex_correction_has_zero_endpoint_and_cannot_overshoot` | FunctionDef | См. реализацию | [L122](../../../../tests/test_chromaseed_kernel.py#L122) |
| `test_residual_bound_survives_correlated_landmarks_and_float32_coefficients` | FunctionDef | См. реализацию | [L136](../../../../tests/test_chromaseed_kernel.py#L136) |

## Все тестовые определения (8)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_nystrom_all_landmarks_matches_independent_weighted_kernel_ridge` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_nystrom_all_landmarks_matches_independent_weighted_kernel_ridge():
    from chromaseed_kernel import gaussian_kernel, nystrom_coefficients

    rng = np.random.default_rng(553)
    x, y, w = rng.normal(size=(22, 5)), rng.normal(size=(22, 3)), rng.uniform(.2, 3., 22)
    k = gaussian_kernel(x, x, .6)
    for alpha in (.1, 1., 10.):
        expected = np.linalg.solve(k + np.diag(alpha / w), y)
        coefficient, info = nystrom_coefficients(k, np.arange(22), y, w, [alpha])
        np.testing.assert_allclose(coefficient[0], expected, rtol=1e-8, atol=1e-9)
        assert info["effective_rank"] == 22
```

</details>

### `test_nystrom_matches_direct_rkhs_regularized_normal_equations` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_nystrom_matches_direct_rkhs_regularized_normal_equations():
    from chromaseed_kernel import gaussian_kernel, nystrom_coefficients

    rng = np.random.default_rng(1961)
    x, y, w = rng.normal(size=(30, 7)), rng.normal(size=(30, 3)), rng.uniform(.3, 2., 30)
    k = gaussian_kernel(x, x, .8)
    idx = np.array([0, 3, 6, 9, 11, 19, 23, 27])
    z = k[:, idx]
    expected = np.linalg.solve(z.T @ (w[:, None] * z) + .7 * k[np.ix_(idx, idx)], z.T @ (w[:, None] * y))
    actual, _ = nystrom_coefficients(k, idx, y, w, [.7])
    np.testing.assert_allclose(actual[0], expected, rtol=1e-10, atol=1e-10)
```

</details>

### `test_median_width_excludes_exact_duplicates_without_distance_cancellation` · L36

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_median_width_excludes_exact_duplicates_without_distance_cancellation():
    from chromaseed_kernel import median_width

    x = np.array([[1e8, 1e8], [1e8, 1e8], [1e8 + 3, 1e8 + 4]])
    assert median_width(x) == pytest.approx(np.sqrt(12.5))
```

</details>

### `test_landmark_prefixes_reconstruct_psd_residual_and_replay` · L44

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('mode', ['random', 'pivot', 'rpchol'])
def test_landmark_prefixes_reconstruct_psd_residual_and_replay(mode):
    from chromaseed_kernel import gaussian_kernel, select_landmarks

    rng = np.random.default_rng(947)
    x, w = rng.normal(size=(35, 5)), rng.uniform(.1, 2., 35)
    k = gaussian_kernel(x, x, 1.)
    idx, factor, residual = select_landmarks(k, w, mode, 12, 17)
    short, _, _ = select_landmarks(k, w, mode, 6, 17)
    np.testing.assert_array_equal(idx[:6], short)
    assert len(np.unique(idx)) == 12
    full = np.sqrt(w)[:, None] * k * np.sqrt(w)[None]
    remaining = full - factor @ factor.T
    np.testing.assert_allclose(np.diag(remaining), residual, rtol=1e-9, atol=1e-12)
    assert np.linalg.eigvalsh(remaining).min() > -1e-10
    assert residual.sum() < np.diag(full).sum()
```

</details>

### `test_projection_diagnostic_bounds_arbitrary_rounded_residuals` · L61

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_projection_diagnostic_bounds_arbitrary_rounded_residuals():
    from chromaseed_kernel import (
        gaussian_kernel,
        projection_coefficients,
        residual_bound,
        residual_diagnostic,
    )

    rng = np.random.default_rng(9321)
    x = rng.normal(size=(25, 6))
    query = np.vstack((x[:8], rng.normal(size=(1000, 6)) * 4))
    teacher = rng.normal(size=(25, 3))
    k = gaussian_kernel(x, x, .9)
    idx = np.arange(8)
    projected, _ = projection_coefficients(k, idx, teacher)
    # Deliberately coarse coefficients destroy exact projection orthogonality.
    projected = np.round(projected, 1)
    norm_sq, residual_at_centers = residual_diagnostic(k, idx, teacher, projected)
    query_kernel = gaussian_kernel(query, x[idx], .9)
    component_bounds = residual_bound(query_kernel, norm_sq, residual_at_centers)
    actual = gaussian_kernel(query, x, .9) @ teacher - query_kernel @ projected
    assert np.all(np.abs(actual) <= component_bounds + 1e-9)
    expected_norm = np.einsum("ic,ij,jc->c", teacher, k, teacher)
    expected_norm += np.einsum("ic,ij,jc->c", projected, k[np.ix_(idx, idx)], projected)
    expected_norm -= 2 * np.einsum("ic,ij,jc->c", projected, k[idx], teacher)
    np.testing.assert_allclose(norm_sq, expected_norm, atol=1e-11)
```

</details>

### `test_adaptive_kernel_computes_only_new_centers_and_honors_forced_exit` · L89

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_adaptive_kernel_computes_only_new_centers_and_honors_forced_exit(monkeypatch):
    import chromaseed_kernel as module

    rng = np.random.default_rng(537)
    centers = rng.normal(size=(8, 36)).astype(np.float32)
    prep = {"x_mean": np.zeros(36, np.float32), "x_std": np.ones(36, np.float32), "y_mean": np.array([50, 4, 10], np.float32), "y_std": np.array([10, 4, 6], np.float32), "width": np.asarray(1., np.float32)}
    levels = []
    k = module.gaussian_kernel(centers, centers, 1.)
    teacher = rng.normal(size=(8, 3)).astype(np.float32)
    for count in (2, 4, 8):
        ids = np.arange(count)
        beta, _ = module.projection_coefficients(k, ids, teacher)
        beta = beta.astype(np.float32)
        q, r = module.residual_diagnostic(k, ids, teacher, beta)
        levels.append((count, beta, q, r))
    payload = module.pack_adaptive(prep, centers, levels, 0.)
    deployed = module.AdaptiveKernel(payload)
    calls = []
    original = module.gaussian_kernel
    def recording_kernel(a, b, width):
        calls.append(len(b))
        return original(a, b, width)
    monkeypatch.setattr(module, "gaussian_kernel", recording_kernel)
    prediction, used, bound, met = deployed.predict_one(centers[0], tolerance=1e6)
    assert used == 2 and sum(calls) == 2 and met
    expected = module.predict_kernel({**prep, "centers": centers[:2], "coefficient": levels[0][1]}, centers[:1])[0]
    np.testing.assert_allclose(prediction, expected, atol=1e-10)
    calls.clear()
    _, used, bound, met = deployed.predict_one(centers[0], tolerance=0.)
    assert used == 8 and sum(calls) == 8 and calls == [2, 2, 4]
    assert bound < 1e-4 and not met
```

</details>

### `test_convex_correction_has_zero_endpoint_and_cannot_overshoot` · L122

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_convex_correction_has_zero_endpoint_and_cannot_overshoot():
    from chromaseed_kernel import convex_blend

    a = np.array([[1., -3., 8.]])
    b = np.array([[-2., 4., 2.]])
    np.testing.assert_array_equal(convex_blend(a, b, 0.), a)
    np.testing.assert_array_equal(convex_blend(a, b, 1.), b)
    for rho in (.25, .5, .75):
        output = convex_blend(a, b, rho)
        assert np.all(output >= np.minimum(a, b)) and np.all(output <= np.maximum(a, b))
    with pytest.raises(ValueError):
        convex_blend(a, b, 1.1)
```

</details>

### `test_residual_bound_survives_correlated_landmarks_and_float32_coefficients` · L136

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_residual_bound_survives_correlated_landmarks_and_float32_coefficients():
    from chromaseed_kernel import (
        exact_coefficients,
        gaussian_kernel,
        projection_coefficients,
        residual_bound,
        residual_diagnostic,
        select_landmarks,
    )

    rng = np.random.default_rng(794)
    projection = rng.normal(size=(3, 36))
    x = rng.normal(size=(170, 3)) @ projection
    x = (x + rng.normal(size=x.shape) * 1e-5).astype(np.float32).astype(np.float64)
    y = rng.normal(size=(170, 3))
    query = (rng.normal(size=(300, 3)) @ projection * 1.5).astype(np.float32).astype(np.float64)
    for width in (.5, 1., 2.):
        k = gaussian_kernel(x, x, width)
        teacher = exact_coefficients(k, y, np.ones(len(x)), [.1])[0].astype(np.float32)
        idx, _, _ = select_landmarks(k, np.ones(len(x)), "rpchol", 128, 17)
        coefficient, _ = projection_coefficients(k, idx, teacher)
        coefficient = coefficient.astype(np.float32)
        q, residual = residual_diagnostic(k, idx, teacher, coefficient)
        km = gaussian_kernel(query, x[idx], width)
        bound = residual_bound(km, q, residual)
        difference = gaussian_kernel(query, x, width) @ teacher.astype(np.float64) - km @ coefficient.astype(np.float64)
        assert np.max(np.abs(difference) - bound) <= 1e-6
```

</details>
