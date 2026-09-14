# `tests/test_chromaseed_hybrid.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_hybrid.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Shared geometry and analytical residual contracts, using synthetic pixels only.

SHA-256 исходника: `c50be30a0831191d25e035f82525c39dc1e5f933670b79fd1e3471403bd3709a`. Строк: **142**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_affine import fit_single as a_single
from chromaseed_affine_reference import qr_ridge
from chromaseed_hybrid import (
    choose,
    fit_bank,
    fit_single,
    model_id,
    package,
    predict,
    shared_basis,
    support,
)
from chromaseed_hybrid_numpy import Predictor
from chromaseed_kernel import gaussian_kernel
from test_chromaseed_projection import toy
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bank` | FunctionDef | См. реализацию | [L28](../../../../tests/test_chromaseed_hybrid.py#L28) |
| `test_support_is_bounded_smooth_kernel_average` | FunctionDef | См. реализацию | [L32](../../../../tests/test_chromaseed_hybrid.py#L32) |
| `test_projected_basis_keeps_requested_original_rows` | FunctionDef | См. реализацию | [L41](../../../../tests/test_chromaseed_hybrid.py#L41) |
| `test_raw_controls_match_frozen_a` | FunctionDef | См. реализацию | [L54](../../../../tests/test_chromaseed_hybrid.py#L54) |
| `test_endpoints_are_exact_payloads` | FunctionDef | См. реализацию | [L66](../../../../tests/test_chromaseed_hybrid.py#L66) |
| `test_portable_queries_and_partition_invariance` | FunctionDef | См. реализацию | [L78](../../../../tests/test_chromaseed_hybrid.py#L78) |
| `test_complete_single_fit_equals_bank` | FunctionDef | См. реализацию | [L99](../../../../tests/test_chromaseed_hybrid.py#L99) |
| `test_single_camera_still_learns_positive_correction` | FunctionDef | См. реализацию | [L107](../../../../tests/test_chromaseed_hybrid.py#L107) |
| `test_residual_solver_matches_augmented_qr` | FunctionDef | См. реализацию | [L115](../../../../tests/test_chromaseed_hybrid.py#L115) |
| `test_predeclared_tie_prefers_smaller_model` | FunctionDef | См. реализацию | [L128](../../../../tests/test_chromaseed_hybrid.py#L128) |
| `test_consumer_rejects_missing_or_invalid_fields` | FunctionDef | См. реализацию | [L138](../../../../tests/test_chromaseed_hybrid.py#L138) |

## Все тестовые определения (10)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_support_is_bounded_smooth_kernel_average` · L32

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_support_is_bounded_smooth_kernel_average():
    k = np.array([[1.0, 0.5, 0.25], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(support(k, 4), [((1 + 0.5 + 0.25) / 3) ** 4, 0])
    assert np.all(support(k * 0.7, 1) <= support(k, 1))
    assert np.all(support(k, 4) <= support(k, 1))
    with pytest.raises(ValueError):
        support(k, 2)
```

</details>

### `test_projected_basis_keeps_requested_original_rows` · L41

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_projected_basis_keeps_requested_original_rows():
    q = np.random.default_rng(11).normal(size=(25, 16))
    ids = np.array([9, 2, 17, 20])
    basis = shared_basis(q, ids, 1.25)
    np.testing.assert_array_equal(basis["ids"], ids)
    np.testing.assert_array_equal(basis["centers"], q[ids].astype(np.float32))
    k = gaussian_kernel(q, q[ids], 1.25)
    np.testing.assert_allclose(basis["z"], k @ basis["whitening"], atol=1e-12)
    np.testing.assert_allclose(
        basis["whitening"].T @ k[ids] @ basis["whitening"], np.eye(4), atol=1e-12
    )
```

</details>

### `test_raw_controls_match_frozen_a` · L54

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_raw_controls_match_frozen_a(bank):
    models, rec = bank
    assert len(models) == 240
    assert rec["new_coefficient_solutions"] == 78 and rec["gram_decompositions"] == 24
    for loss in ("norm", "perceptual"):
        expected, _ = a_single(*toy(), loss + "_joint_soft", 17, 0.1, 0.0, rank=4)
        actual = models[model_id(loss, 17, "raw")]
        assert set(actual) == set(expected)
        for k in expected:
            np.testing.assert_array_equal(actual[k], expected[k])
```

</details>

### `test_endpoints_are_exact_payloads` · L66

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_endpoints_are_exact_payloads(bank):
    models, _ = bank
    raw = models[model_id("norm", 17, "raw")]
    proj = models[model_id("norm", 17, "projected", 1.0)]
    zero = package(raw, proj, "support", 0.0, 4)
    one = package(raw, proj, "blend", 1.0, 0)
    for actual, expected in ((zero, raw), (one, proj)):
        assert set(actual) == set(expected)
        for k in actual:
            np.testing.assert_array_equal(actual[k], expected[k])
```

</details>

### `test_portable_queries_and_partition_invariance` · L78

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_portable_queries_and_partition_invariance(bank):
    models, _ = bank
    x = toy()[0][3:8]
    for m in models.values():
        p = Predictor(m)
        np.testing.assert_allclose([p(v) for v in x], predict(m, x), atol=2e-8, rtol=0)
        np.testing.assert_allclose(predict(m, x[:2]), predict(m, x)[:2], atol=2e-8, rtol=0)
        assert p.cached_array_bytes > 0
```

</details>

### `test_complete_single_fit_equals_bank` · L99

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('kind,alpha,rho,power', [('raw', 0.1, 0.0, 0), ('projected', 1.0, 1.0, 0), ('blend', 10.0, 0.25, 0), ('uniform', 0.1, 0.5, 0), ('support', 1.0, 1.0, 1), ('support', 10.0, 0.5, 4)])
def test_complete_single_fit_equals_bank(bank, kind, alpha, rho, power):
    expected = bank[0][model_id("perceptual", 29, kind, alpha, rho, power)]
    actual, _ = fit_single(*toy(), "perceptual", 29, kind, alpha, rho, power, rank=4)
    assert set(expected) == set(actual)
    for k in expected:
        np.testing.assert_array_equal(expected[k], actual[k])
```

</details>

### `test_single_camera_still_learns_positive_correction` · L107

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_single_camera_still_learns_positive_correction():
    models, _ = fit_bank(*toy(False), rank=4)
    raw = models[model_id("perceptual", 17, "raw")]
    m = models[model_id("perceptual", 17, "support", 0.1, 0.5, 4)]
    assert "gate_beta" not in m and "hybrid_mode" in m
    assert np.max(np.abs(predict(m, toy(False)[0]) - predict(raw, toy(False)[0]))) > 1e-5
```

</details>

### `test_residual_solver_matches_augmented_qr` · L115

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_residual_solver_matches_augmented_qr():
    from chromaseed_affine import gram_stats, solve_stats

    rng = np.random.default_rng(902)
    d, r, w = rng.normal(size=(50, 9)), rng.normal(size=(50, 3)), rng.uniform(0.1, 3, 50)
    h = rng.uniform(0.1, 0.9, 50) ** 4
    metric = np.array([[1.0, 0.15, 0], [0.15, 2.0, 0.1], [0, 0.1, 0.7]])
    sol, _ = solve_stats(gram_stats(h[:, None] * d, r, w), {"test": metric}, (0.1,))
    np.testing.assert_allclose(
        sol[("test", 0.1)], qr_ridge(h[:, None] * d, r, w, metric, 0.1), atol=1e-11
    )
```

</details>

### `test_predeclared_tie_prefers_smaller_model` · L128

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_predeclared_tie_prefers_smaller_model():
    rows = [
        dict(clean=4.0, numeric_bytes=27000, p90=7.0, alpha=0.1, rho=0.5, power=4),
        dict(clean=4.000001, numeric_bytes=21000, p90=7.1, alpha=0.1, rho=0.0, power=0),
    ]
    assert choose(rows) is rows[1]
    rows[1]["clean"] = 4.001
    assert choose(rows) is rows[0]
```

</details>

### `test_consumer_rejects_missing_or_invalid_fields` · L138

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_consumer_rejects_missing_or_invalid_fields(bank):
    m = dict(bank[0][model_id("norm", 17, "support", 0.1, 0.5, 4)])
    del m["latent_width"]
    with pytest.raises(ValueError):
        Predictor(m)
```

</details>
