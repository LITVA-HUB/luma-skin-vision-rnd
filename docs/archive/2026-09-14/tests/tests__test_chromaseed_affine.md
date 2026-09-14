# `tests/test_chromaseed_affine.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_affine.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Mathematical and protocol invariants for weighted affine-augmentation learning.

SHA-256 исходника: `9c6fd0370c12893dfcf3ff659ef67bc759853b7147226fc21f55e82f220fe012`. Строк: **191**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_affine import (
    FAMILIES,
    choose,
    combine,
    design_matrix,
    export_model,
    fit_bank,
    fit_single,
    gram_stats,
    model_id,
    predict,
    solve_stats,
    weighted_copies,
)
from chromaseed_perceptual_reference import augmented_svd
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `problem` | FunctionDef | См. реализацию | [L27](../../../../tests/test_chromaseed_affine.py#L27) |
| `test_copy_mass_and_gram_equal_explicit_stacked_design` | FunctionDef | См. реализацию | [L37](../../../../tests/test_chromaseed_affine.py#L37) |
| `test_coupled_stats_solver_matches_independent_augmented_svd` | FunctionDef | См. реализацию | [L52](../../../../tests/test_chromaseed_affine.py#L52) |
| `test_zero_fraction_ignores_augmented_stats` | FunctionDef | См. реализацию | [L63](../../../../tests/test_chromaseed_affine.py#L63) |
| `test_single_camera_joint_design_is_static_and_soft_is_continuous` | FunctionDef | См. реализацию | [L71](../../../../tests/test_chromaseed_affine.py#L71) |
| `test_guard_keeps_clean_budget_and_prefers_weaker_augmentation_on_tie` | FunctionDef | См. реализацию | [L81](../../../../tests/test_chromaseed_affine.py#L81) |
| `test_invalid_augmentation_weights_rejected` | FunctionDef | См. реализацию | [L97](../../../../tests/test_chromaseed_affine.py#L97) |
| `test_exported_joint_readout_matches_analytical_function` | FunctionDef | См. реализацию | [L105](../../../../tests/test_chromaseed_affine.py#L105) |
| `test_constant_predictor_remains_constant_and_returns_independent_values` | FunctionDef | См. реализацию | [L130](../../../../tests/test_chromaseed_affine.py#L130) |
| `toy` | FunctionDef | См. реализацию | [L139](../../../../tests/test_chromaseed_affine.py#L139) |
| `test_single_camera_bank_joint_payloads_are_exact_static_aliases` | FunctionDef | См. реализацию | [L164](../../../../tests/test_chromaseed_affine.py#L164) |
| `test_complete_single_fits_reproduce_shared_bank_payloads` | FunctionDef | См. реализацию | [L182](../../../../tests/test_chromaseed_affine.py#L182) |

## Все тестовые определения (10)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_copy_mass_and_gram_equal_explicit_stacked_design` · L37

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('eta', [0, 0.25, 0.75])
def test_copy_mass_and_gram_equal_explicit_stacked_design(eta):
    z, y, w, aug = problem()
    weights = weighted_copies(w, eta)
    np.testing.assert_allclose(weights.sum(0), w, rtol=0, atol=5e-16)
    original = gram_stats(z, y, w)
    augmented = tuple(np.mean([gram_stats(a, y, w)[i] for a in aug], axis=0) for i in range(3))
    mixed = combine(original, augmented, eta)
    stacked = np.concatenate([z[None], aug])
    reference = gram_stats(stacked.reshape(-1, z.shape[1]), np.tile(y, (17, 1)), weights.ravel())
    for actual, expected in zip(mixed, reference, strict=True):
        # The stacked BLAS reduction sums221 rows in a different order; fsum
        # confirms the observed3.84e-15 relative discrepancy is accumulation noise.
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=2e-14)
```

</details>

### `test_coupled_stats_solver_matches_independent_augmented_svd` · L52

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_coupled_stats_solver_matches_independent_augmented_svd():
    z, y, w, _ = problem()
    metric = np.array([[2.0, 0.1, -0.2], [0.1, 0.8, 0.12], [-0.2, 0.12, 1.3]])
    solved, info = solve_stats(gram_stats(z, y, w), {"m": metric}, (0.1, 1.0, 10.0))
    for a in (0.1, 1.0, 10.0):
        expected = augmented_svd(z, y, np.broadcast_to(metric, (len(z), 3, 3)), w, a)
        np.testing.assert_allclose(solved[("m", a)], expected, atol=2e-13, rtol=0)
    assert max(r["objective_minus_zero"] for r in info["solutions"]) <= 0
    assert max(r["normal_residual"] for r in info["solutions"]) < 1e-12
```

</details>

### `test_zero_fraction_ignores_augmented_stats` · L63

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_zero_fraction_ignores_augmented_stats():
    z, y, w, _ = problem()
    stats = gram_stats(z, y, w)
    corrupt = tuple(np.full_like(a, np.nan) for a in stats)
    for actual, expected in zip(combine(stats, corrupt, 0), stats, strict=True):
        np.testing.assert_array_equal(actual, expected)
```

</details>

### `test_single_camera_joint_design_is_static_and_soft_is_continuous` · L71

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_single_camera_joint_design_is_static_and_soft_is_continuous():
    z, _, _, _ = problem()
    np.testing.assert_array_equal(design_matrix(z, None, True), z)
    for sign in (-1e-5, 0.0, 1e-5):
        s = np.full(len(z), sign)
        d = design_matrix(z, s, True)
        np.testing.assert_array_equal(d[:, :5], z)
        np.testing.assert_allclose(d[:, 5:], sign * z)
```

</details>

### `test_guard_keeps_clean_budget_and_prefers_weaker_augmentation_on_tie` · L81

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_guard_keeps_clean_budget_and_prefers_weaker_augmentation_on_tie():
    candidates = [
        dict(alpha=0.1, eta=0.0, clean=5.0, robust=6.0),
        dict(alpha=1.0, eta=0.25, clean=5.049, robust=5.5),
        dict(alpha=10.0, eta=0.75, clean=5.051, robust=5.0),
    ]
    assert choose(candidates, "clean")["eta"] == 0
    assert choose(candidates, "guarded")["eta"] == 0.25
    tied = [
        dict(alpha=a, eta=e, clean=5.0, robust=6.0) for a in (0.1, 1.0, 10.0) for e in (0.0, 0.25)
    ]
    for policy in ("clean", "guarded"):
        assert choose(tied, policy)["eta"] == 0
        assert choose(tied, policy)["alpha"] == 10
```

</details>

### `test_invalid_augmentation_weights_rejected` · L97

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_augmentation_weights_rejected():
    for eta in (-1, 1, np.nan):
        with pytest.raises(ValueError):
            weighted_copies(np.ones(3), eta)
    with pytest.raises(ValueError):
        weighted_copies(np.array([1.0, 0]), 0.25)
```

</details>

### `test_exported_joint_readout_matches_analytical_function` · L105

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_exported_joint_readout_matches_analytical_function():
    x = np.random.default_rng(99).uniform(0.1, 0.8, (5, 36)).astype(np.float32)
    prep = dict(
        x_mean=np.zeros(36, np.float32),
        x_std=np.ones(36, np.float32),
        y_mean=np.array([50, 10, 10], np.float32),
        y_std=np.array([10, 5, 5], np.float32),
    )
    state = dict(prep=prep, normalized=x.astype(np.float64))
    basis = dict(ids=np.arange(3), whitening=np.eye(3), width=1.0)
    theta = np.arange(18).reshape(6, 3) / 8
    beta = np.r_[-1.0, 2.0, np.zeros(35)].astype(np.float32)
    m = export_model(state, basis, theta, beta, True)
    k = np.exp(
        -0.5 * ((x.astype(np.float64)[:, None] - x[None, :3].astype(np.float64)) ** 2).sum(-1) / 36
    )
    s = np.clip(2 * x[:, 0].astype(np.float64) - 1, -1, 1)
    expected = (k @ theta[:3] + s[:, None] * (k @ theta[3:])) * prep["y_std"] + prep["y_mean"]
    np.testing.assert_allclose(predict(m, x), expected, atol=1e-10, rtol=0)
    assert (
        sum(v.nbytes for v in m.values())
        == (36 * 2 + 3 * 2 + 3 * 36 + 3 * 3 + 1 + 3 * 3 + 37 + 1) * 4 + 1
    )
```

</details>

### `test_constant_predictor_remains_constant_and_returns_independent_values` · L130

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_constant_predictor_remains_constant_and_returns_independent_values():
    model = dict(constant_lab=np.array([55.0, 7.0, 12.0], np.float32))
    x = np.random.default_rng(7).normal(size=(8, 36))
    out = predict(model, x)
    np.testing.assert_array_equal(out, np.tile(model["constant_lab"], (8, 1)))
    out[0, 0] = 0
    assert model["constant_lab"][0] == 55
```

</details>

### `test_single_camera_bank_joint_payloads_are_exact_static_aliases` · L164

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_single_camera_bank_joint_payloads_are_exact_static_aliases():
    models, receipt = fit_bank(*toy(True), rank=4)
    assert receipt["gate_fits"] == 0 and receipt["new_coefficient_solutions"] == 72
    for name, rec in receipt["models"].items():
        if rec["fallback"]:
            baseline = models[
                model_id(
                    rec["family"].replace("joint_soft", "static"),
                    rec["seed"],
                    rec["alpha"],
                    rec["eta"],
                )
            ]
            assert set(models[name]) == set(baseline)
            for key in baseline:
                np.testing.assert_array_equal(models[name][key], baseline[key])
```

</details>

### `test_complete_single_fits_reproduce_shared_bank_payloads` · L182

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_complete_single_fits_reproduce_shared_bank_payloads():
    data = toy(False)
    models, receipt = fit_bank(*data, rank=4)
    assert receipt["gate_fits"] == 1 and receipt["new_coefficient_solutions"] == 144
    for family in FAMILIES:
        fitted, _ = fit_single(*data, family, 17, 1.0, 0.25, rank=4)
        expected = models[model_id(family, 17, 1.0, 0.25)]
        assert set(fitted) == set(expected)
        for key in expected:
            np.testing.assert_array_equal(fitted[key], expected[key])
```

</details>
