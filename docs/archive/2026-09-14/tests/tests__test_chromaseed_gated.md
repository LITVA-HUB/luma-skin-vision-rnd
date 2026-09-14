# `tests/test_chromaseed_gated.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_gated.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `8e049497bd4509bde9566af08b91615d37968a6648148ad10e2250eb39c61043`. Строк: **119**.

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
| `sample` | FunctionDef | См. реализацию | [L10](../../../../tests/test_chromaseed_gated.py#L10) |
| `test_zero_and_base_payloads_preserve_frozen_p` | FunctionDef | См. реализацию | [L20](../../../../tests/test_chromaseed_gated.py#L20) |
| `test_one_camera_routed_fallback_is_exact_but_uniform_can_fit` | FunctionDef | См. реализацию | [L36](../../../../tests/test_chromaseed_gated.py#L36) |
| `test_coupled_residual_bank_matches_augmented_svd` | FunctionDef | См. реализацию | [L52](../../../../tests/test_chromaseed_gated.py#L52) |
| `test_gate_clipping_and_exact_hard_boundary` | FunctionDef | См. реализацию | [L70](../../../../tests/test_chromaseed_gated.py#L70) |
| `test_static_collapse_and_dynamic_predictor_share_one_basis` | FunctionDef | См. реализацию | [L80](../../../../tests/test_chromaseed_gated.py#L80) |
| `test_registered_candidate_grid_and_invalid_settings` | FunctionDef | См. реализацию | [L112](../../../../tests/test_chromaseed_gated.py#L112) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_zero_and_base_payloads_preserve_frozen_p` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('base,old', [('norm', 'norm_mse'), ('perceptual', 'constant_de2')])
def test_zero_and_base_payloads_preserve_frozen_p(base, old):
    from chromaseed_gated import fit_single
    from chromaseed_perceptual import fit_single as pfit
    from skin_local_search_train import weights_for

    x, y, p, s, c = sample()
    w = weights_for(p, s)
    expected, _ = pfit(x, y, w, old, 17, 1, 0, 0 if base == "norm" else 1, rank=16)
    for route in ("base", "uniform", "soft", "hard"):
        actual, _ = fit_single(x, y, p, s, c, f"{base}_{route}", 17, 0.1, 0.0, rank=16)
        assert set(actual) == set(expected)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
```

</details>

### `test_one_camera_routed_fallback_is_exact_but_uniform_can_fit` · L36

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('route', ['soft', 'hard'])
def test_one_camera_routed_fallback_is_exact_but_uniform_can_fit(route):
    from chromaseed_gated import fit_single

    x, y, p, s, c = sample()
    c[:] = "SLR"
    expected, _ = fit_single(x, y, p, s, c, "norm_base", 17, 0.1, 0.0, rank=16)
    actual, info = fit_single(x, y, p, s, c, f"norm_{route}", 17, 0.1, 1.0, rank=16)
    assert info["fallback"] == "single_camera" and info["residual_solutions"] == 0
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    ordinary, info = fit_single(x, y, p, s, c, "norm_uniform", 17, 0.1, 1.0, rank=16)
    assert info["residual_solutions"] == 1 and not np.array_equal(
        ordinary["coefficient"], expected["coefficient"]
    )
```

</details>

### `test_coupled_residual_bank_matches_augmented_svd` · L52

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_coupled_residual_bank_matches_augmented_svd():
    from chromaseed_gated import residual_solutions
    from chromaseed_perceptual_reference import augmented_svd

    rng = np.random.default_rng(312)
    z = rng.normal(size=(49, 9))
    z[:, 8] = z[:, 0] + 1e-7 * z[:, 8]
    y = rng.normal(size=(49, 3))
    w = rng.uniform(0.2, 2.0, 49)
    a = rng.normal(size=(3, 3))
    g = a @ a.T + 0.2 * np.eye(3)
    theta, info = residual_solutions(z, y, g, w, (0.1, 1.0, 10.0))
    for alpha in theta:
        reference = augmented_svd(z, y, np.broadcast_to(g, (49, 3, 3)), w, alpha)
        np.testing.assert_allclose(z @ theta[alpha], z @ reference, atol=1e-10, rtol=0)
    assert max(d["objective_minus_zero"] for d in info["solutions"]) <= 1e-10
```

</details>

### `test_gate_clipping_and_exact_hard_boundary` · L70

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_gate_clipping_and_exact_hard_boundary():
    from chromaseed_gated import gate_value

    z = np.zeros((5, 36))
    z[:, 0] = [-2.0, -0.1, 0.0, 0.1, 2.0]
    beta = np.r_[0.0, 1.0, np.zeros(35)].astype(np.float32)
    np.testing.assert_array_equal(gate_value(z, beta, "soft"), [-1.0, -0.1, 0.0, 0.1, 1.0])
    np.testing.assert_array_equal(gate_value(z, beta, "hard"), [-1.0, -1.0, 1.0, 1.0, 1.0])
```

</details>

### `test_static_collapse_and_dynamic_predictor_share_one_basis` · L80

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_static_collapse_and_dynamic_predictor_share_one_basis():
    from chromaseed_gated import package_model, predict
    from chromaseed_kernel import coordinates, gaussian_kernel, predict_kernel

    x, _, _, _, _ = sample()
    base = dict(
        x_mean=np.zeros(36, np.float32),
        x_std=np.ones(36, np.float32),
        y_mean=np.zeros(3, np.float32),
        y_std=np.ones(3, np.float32),
        centers=x[:7],
        width=np.array(1.0, np.float32),
        coefficient=np.ones((7, 3), np.float32),
    )
    correction = np.ones((7, 3), np.float32) * 0.25
    beta = np.r_[0.0, 1.0, np.zeros(35)].astype(np.float32)
    static = package_model(base, correction, None, "uniform", 0.5)
    assert set(static) == set(base)
    np.testing.assert_array_equal(static["coefficient"], np.full((7, 3), 1.125, np.float32))
    dynamic = package_model(base, correction, beta, "hard", 0.5)
    z = coordinates(base, x)
    k = gaussian_kernel(z, base["centers"], 1.0)
    expected = predict_kernel(base, x) + 0.5 * np.where(z[:, 0] >= 0, 1.0, -1.0)[:, None] * (
        k @ correction
    )
    np.testing.assert_allclose(predict(dynamic, x), expected, atol=1e-12, rtol=0)
    assert (
        sum(v.nbytes for v in dynamic.values()) - sum(v.nbytes for v in base.values())
        == 7 * 3 * 4 + 37 * 4 + 4 + 1
    )
```

</details>

### `test_registered_candidate_grid_and_invalid_settings` · L112

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_registered_candidate_grid_and_invalid_settings():
    from chromaseed_gated import FAMILIES, candidates, fit_single

    assert len(FAMILIES) == 8
    assert sum(len(candidates(f)) for f in FAMILIES) == 62
    x, y, p, s, c = sample()
    with pytest.raises(ValueError):
        fit_single(x, y, p, s, c, "norm_soft", 17, -1.0, 1.0, rank=16)
```

</details>
