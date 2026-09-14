# `tests/test_chromaseed_neural_readout.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_neural_readout.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Numerical oracles for unpenalized-bias compact neural readouts.

SHA-256 исходника: `aa1d876bad80388ab39f3cc8f81400bfea0e920d5b694c532bc35a3a906304bc`. Строк: **182**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_gaussian import export, initialize, predict, prepare
from chromaseed_gaussian_numpy import Predictor
from chromaseed_neural_readout import fit_head, fit_single, representation_bank, solve
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fixture` | FunctionDef | См. реализацию | [L15](../../../../tests/test_chromaseed_neural_readout.py#L15) |
| `augmented` | FunctionDef | См. реализацию | [L24](../../../../tests/test_chromaseed_neural_readout.py#L24) |
| `test_unpenalized_intercept_matches_scalar_augmented_lstsq` | FunctionDef | См. реализацию | [L36](../../../../tests/test_chromaseed_neural_readout.py#L36) |
| `test_correlated_output_metric_matches_independent_augmented_lstsq` | FunctionDef | См. реализацию | [L52](../../../../tests/test_chromaseed_neural_readout.py#L52) |
| `test_folded_network_preserves_representation_and_actual_consumer` | FunctionDef | См. реализацию | [L65](../../../../tests/test_chromaseed_neural_readout.py#L65) |
| `test_all_dead_hidden_channels_fit_an_unpenalized_constant` | FunctionDef | См. реализацию | [L85](../../../../tests/test_chromaseed_neural_readout.py#L85) |
| `test_mean3_ignores_finite_other_features` | FunctionDef | См. реализацию | [L99](../../../../tests/test_chromaseed_neural_readout.py#L99) |
| `test_invalid_fit_inputs_are_rejected` | FunctionDef | См. реализацию | [L110](../../../../tests/test_chromaseed_neural_readout.py#L110) |
| `test_fixed_representation_banks_match_single_full_construction` | FunctionDef | См. реализацию | [L127](../../../../tests/test_chromaseed_neural_readout.py#L127) |
| `test_independent_qr_head_matches_native_predictions` | FunctionDef | См. реализацию | [L146](../../../../tests/test_chromaseed_neural_readout.py#L146) |
| `test_selection_ties_prefer_declared_alpha_and_shorter_representation` | FunctionDef | См. реализацию | [L159](../../../../tests/test_chromaseed_neural_readout.py#L159) |
| `test_registered_representation_prefixes_match_independent_scalar_training` | FunctionDef | См. реализацию | [L169](../../../../tests/test_chromaseed_neural_readout.py#L169) |

## Все тестовые определения (10)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_unpenalized_intercept_matches_scalar_augmented_lstsq` · L36

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_unpenalized_intercept_matches_scalar_augmented_lstsq():
    z = np.column_stack((np.ones(7), np.linspace(-2, 3, 7)))
    target = np.column_stack((3 + z[:, 1], -5 + 2 * z[:, 1], 8 - z[:, 1]))
    w = np.arange(1, 8, dtype=float)
    alpha = 4.0
    expected = augmented(z, target, np.broadcast_to(np.eye(3), (7, 3, 3)), w, alpha)
    beta, residual = solve(z, target, None, w, alpha)
    np.testing.assert_allclose(beta, expected, atol=1e-12)
    assert residual < 1e-12
    shift = np.array([100, -70, 45])
    shifted, _ = solve(z, target + shift, None, w, alpha)
    np.testing.assert_allclose(shifted[0] - beta[0], shift, atol=1e-12)
    np.testing.assert_allclose(shifted[1:], beta[1:], atol=1e-12)
```

</details>

### `test_correlated_output_metric_matches_independent_augmented_lstsq` · L52

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0])
def test_correlated_output_metric_matches_independent_augmented_lstsq(alpha):
    rng = np.random.default_rng(190)
    z = np.column_stack((np.ones(19), rng.normal(size=(19, 5))))
    t, w = rng.normal(size=(19, 3)), rng.uniform(0.2, 2, 19)
    c = rng.normal(size=(19, 3, 3))
    metric = c @ c.transpose(0, 2, 1) + np.eye(3)[None] * 0.2
    beta, residual = solve(z, t, metric, w, alpha)
    np.testing.assert_allclose(beta, augmented(z, t, metric, w, alpha), atol=2e-12)
    assert residual < 1e-12
```

</details>

### `test_folded_network_preserves_representation_and_actual_consumer` · L65

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('group', ['raw36', 'mean3'])
@pytest.mark.parametrize('family', ['norm', 'perceptual'])
def test_folded_network_preserves_representation_and_actual_consumer(group, family):
    source, x, y, w = fixture(group)
    original = {k: v.copy() for k, v in source.items()}
    model, info = fit_head(source, x, y, w, family, 0.1)
    for k in source:
        np.testing.assert_array_equal(source[k], original[k])
        if k not in ("w2", "b2"):
            np.testing.assert_array_equal(model[k], source[k])
    assert set(model) == set(source)
    assert sum(v.nbytes for v in model.values()) == sum(v.nbytes for v in source.values())
    assert all(v.dtype == original[k].dtype for k, v in model.items())
    assert info["max_folded_lab_drift"] < 0.001
    assert info["normal_residual"] < 1e-10
    consumer = Predictor(model)
    actual = np.array([consumer(row) for row in x])
    np.testing.assert_allclose(actual, predict(model, x), atol=1e-10)
    assert np.isfinite(actual).all()
```

</details>

### `test_all_dead_hidden_channels_fit_an_unpenalized_constant` · L85

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family', ['norm', 'perceptual'])
def test_all_dead_hidden_channels_fit_an_unpenalized_constant(family):
    source, x, y, w = fixture()
    source["w1"][:] = 0
    source["b1"][:] = -1
    model, info = fit_head(source, x, y, w, family, 10)
    assert info["hidden_constant_columns"] == 8
    np.testing.assert_allclose(model["w2"], 0, atol=1e-12)
    assert np.isfinite(predict(model, x)).all()
    if family == "norm":
        np.testing.assert_allclose(
            predict(model, x), np.broadcast_to(np.average(y, weights=w, axis=0), y.shape), atol=2e-6
        )
```

</details>

### `test_mean3_ignores_finite_other_features` · L99

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_mean3_ignores_finite_other_features():
    source, x, y, w = fixture("mean3")
    model, _ = fit_head(source, x, y, w, "norm", 1)
    changed = x.copy()
    changed[:, :27] = 12
    changed[:, 30:] = -8
    consumer = Predictor(model)
    np.testing.assert_array_equal([consumer(row) for row in x], [consumer(row) for row in changed])
```

</details>

### `test_invalid_fit_inputs_are_rejected` · L110

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('bad', ['alpha', 'weight', 'family', 'shape', 'nan'])
def test_invalid_fit_inputs_are_rejected(bad):
    source, x, y, w = fixture()
    alpha, family = 1, "norm"
    if bad == "alpha":
        alpha = 0
    elif bad == "weight":
        w[0] = -1
    elif bad == "family":
        family = "unknown"
    elif bad == "shape":
        y = y[:, :2]
    else:
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_head(source, x, y, w, family, alpha)
```

</details>

### `test_fixed_representation_banks_match_single_full_construction` · L127

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fixed_representation_banks_match_single_full_construction():
    _, x, y, _ = fixture("mean3", n=9)
    person, site = np.repeat(np.arange(3), 3), np.tile(np.arange(3), 3)
    from skin_local_search_train import weights_for

    bank, trajectories = representation_bank(
        x, y, weights_for(person, site), "mean3", seeds=(17,), epochs=(1, 4)
    )
    assert len(bank) == 5 and len(trajectories) == 2
    for basis in ("random", "adam_e1", "adam_e4", "tagi_full3_e1", "tagi_full3_e4"):
        expected, _ = fit_head(bank[(basis, 17)], x, y, weights_for(person, site), "norm", 1)
        actual, _ = fit_single(x, y, person, site, "mean3", basis, "norm", 1, 17)
        for k in expected:
            np.testing.assert_array_equal(expected[k], actual[k])
        assert sum(v.nbytes for v in actual.values()) == 1855
```

</details>

### `test_independent_qr_head_matches_native_predictions` · L146

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('group', ['raw36', 'mean3'])
@pytest.mark.parametrize('family', ['norm', 'perceptual'])
def test_independent_qr_head_matches_native_predictions(group, family):
    from chromaseed_neural_readout_reference import refit

    source, x, y, w = fixture(group, n=43, hidden=64)
    fitted, _ = fit_head(source, x, y, w, family, 0.1)
    independent, info = refit(source, x, y, w, family, 0.1)
    np.testing.assert_allclose(predict(fitted, x), predict(independent, x), atol=0.001, rtol=0)
    assert info["rank"] == 65 * (3 if family == "perceptual" else 1)
    for k in source:
        if k not in ("w2", "b2"):
            np.testing.assert_array_equal(fitted[k], independent[k])
```

</details>

### `test_selection_ties_prefer_declared_alpha_and_shorter_representation` · L159

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_ties_prefer_declared_alpha_and_shorter_representation():
    from chromaseed_neural_readout import choose, choose_policy

    a = [dict(clean=2, p90=4, alpha_index=i) for i in (2, 1, 0)]
    assert choose(a)["alpha_index"] == 0
    b = [dict(clean=2, p90=4, basis=v) for v in ("tagi_full3_e16", "adam_e1", "random")]
    assert choose_policy(b)["basis"] == "random"
```

</details>

### `test_registered_representation_prefixes_match_independent_scalar_training` · L169

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('group', ['raw36', 'mean3'])
def test_registered_representation_prefixes_match_independent_scalar_training(group):
    from chromaseed_gaussian_reference import train_reference

    _, x, y, weights = fixture(group, n=9)
    bank, _ = representation_bank(x, y, weights, group, seeds=(17,))
    for method, parameter in (("adam", 0.001), ("tagi_full3", 1.0)):
        independent, _ = train_reference(
            x, y, weights, group, method, parameter, 17, checkpoints=(1, 4, 16)
        )
        for epoch in (1, 4, 16):
            for k, value in independent[epoch].items():
                np.testing.assert_allclose(
                    bank[(f"{method}_e{epoch}", 17)][k], value, atol=2e-6, rtol=2e-6
                )
```

</details>
