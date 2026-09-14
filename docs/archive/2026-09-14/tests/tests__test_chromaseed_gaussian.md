# `tests/test_chromaseed_gaussian.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_gaussian.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Numerical oracles for local Gaussian learning and a matched Adam network.

SHA-256 исходника: `8b0dc4f8ab765ca14a8dfc6c9822c6fa8b565cbeb8a36390e533a75e97c14f32`. Строк: **291**.

## Зависимости

```python
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from chromaseed_gaussian import (
    KEYS,
    PARAMETERS,
    adam_step,
    choose,
    export,
    gaussian_step,
    initialize,
    predict,
    prepare,
    train_block,
)
from chromaseed_gaussian_numpy import Predictor
from chromaseed_gaussian_reference import train_reference
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bank` | FunctionDef | См. реализацию | [L29](../../../../tests/test_chromaseed_gaussian.py#L29) |
| `dense_oracle` | FunctionDef | Autograd builds a full parameter/output covariance, independent of local gains. | [L37](../../../../tests/test_chromaseed_gaussian.py#L37) |
| `test_gaussian_update_matches_dense_parameter_conditioning` | FunctionDef | См. реализацию | [L68](../../../../tests/test_chromaseed_gaussian.py#L68) |
| `test_scalar_linear_gaussian_readout` | FunctionDef | См. реализацию | [L87](../../../../tests/test_chromaseed_gaussian.py#L87) |
| `test_correlated_outputs_expose_diagonal_variance_floor` | FunctionDef | См. реализацию | [L108](../../../../tests/test_chromaseed_gaussian.py#L108) |
| `test_adam_matches_torch_autograd_for_multiple_steps` | FunctionDef | См. реализацию | [L123](../../../../tests/test_chromaseed_gaussian.py#L123) |
| `synthetic` | FunctionDef | См. реализацию | [L149](../../../../tests/test_chromaseed_gaussian.py#L149) |
| `test_independent_trajectories_and_checkpoint_prefix` | FunctionDef | См. реализацию | [L158](../../../../tests/test_chromaseed_gaussian.py#L158) |
| `test_consumer_normalization_and_numeric_capacity` | FunctionDef | См. реализацию | [L177](../../../../tests/test_chromaseed_gaussian.py#L177) |
| `test_seed_initialization_and_validation` | FunctionDef | См. реализацию | [L204](../../../../tests/test_chromaseed_gaussian.py#L204) |
| `test_candidate_ties_prefer_fewer_epochs_then_declared_order` | FunctionDef | См. реализацию | [L235](../../../../tests/test_chromaseed_gaussian.py#L235) |
| `test_independent_scalar_training_trace` | FunctionDef | См. реализацию | [L245](../../../../tests/test_chromaseed_gaussian.py#L245) |
| `test_full_width_long_schedule_is_finite_and_counted` | FunctionDef | См. реализацию | [L269](../../../../tests/test_chromaseed_gaussian.py#L269) |

## Все тестовые определения (10)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_gaussian_update_matches_dense_parameter_conditioning` · L68

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('kind', ['tagi_diag', 'tagi_full3'])
def test_gaussian_update_matches_dense_parameter_conditioning(kind):
    for seed in (17, 29, 43):
        mean, var = bank(seed=seed)
        x, y = np.array([0.6, -0.4]), np.array([0.3, -0.2, 0.8])
        expected_mean, expected_var, pre = dense_oracle(mean, var, x, y, 0.7, 0.3, kind)
        stats = gaussian_step(mean, var, x[None], y[None], np.array([0.7]), np.array([0.3]), kind)
        np.testing.assert_allclose(
            np.concatenate([mean[k][0].ravel() for k in KEYS]),
            expected_mean,
            atol=2e-12,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            np.concatenate([var[k][0].ravel() for k in KEYS]), expected_var, atol=2e-12, rtol=2e-12
        )
        assert int(stats["negative"][0]) == int(np.sum(pre < 0))
        assert int(stats["floored"][0]) == int(np.sum(pre < 1e-12))
```

</details>

### `test_scalar_linear_gaussian_readout` · L87

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_scalar_linear_gaussian_readout():
    mean, var = bank(d=1, hidden=2)
    mean["w1"][:] = np.array([[[1.0], [2.0]]])
    mean["b1"][:] = 0.5
    var["w1"][:] = var["b1"][:] = 0
    x, target, sigma = np.array([0.8]), np.array([0.4, -0.2, 1.0]), 0.3
    features = np.array([1.3, 2.1, 1.0])
    old_mu = np.column_stack((mean["w2"][0], mean["b2"][0]))
    old_var = np.column_stack((var["w2"][0], var["b2"][0]))
    s = sigma**2 + np.sum(old_var * features**2, axis=1)
    wanted_mu = old_mu + old_var * features * ((target - old_mu @ features) / s)[:, None]
    wanted_var = old_var - (old_var * features) ** 2 / s[:, None]
    gaussian_step(mean, var, x[None], target[None], np.ones(1), np.array([sigma]), "tagi_full3")
    np.testing.assert_allclose(
        np.column_stack((mean["w2"][0], mean["b2"][0])), wanted_mu, atol=1e-12
    )
    np.testing.assert_allclose(
        np.column_stack((var["w2"][0], var["b2"][0])), wanted_var, atol=1e-12
    )
```

</details>

### `test_correlated_outputs_expose_diagonal_variance_floor` · L108

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_correlated_outputs_expose_diagonal_variance_floor():
    mean, var = bank(d=1, hidden=1)
    mean["w1"][:] = mean["w2"][:] = 1
    mean["b1"][:] = mean["b2"][:] = 0
    var["w1"][:] = 1
    for k in ("b1", "w2", "b2"):
        var[k][:] = 0
    diag_m, diag_v = copy.deepcopy(mean), copy.deepcopy(var)
    args = (np.ones((1, 1)), np.ones((1, 3)), np.ones(1), np.array([0.001]))
    ds = gaussian_step(diag_m, diag_v, *args, "tagi_diag")
    fs = gaussian_step(mean, var, *args, "tagi_full3")
    assert ds["negative"][0] >= 1 and diag_v["w1"][0, 0, 0] == 1e-12
    assert fs["negative"][0] == 0 and var["w1"][0, 0, 0] > 1e-8
```

</details>

### `test_adam_matches_torch_autograd_for_multiple_steps` · L123

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_adam_matches_torch_autograd_for_multiple_steps():
    mean, _ = bank(d=3, hidden=5)
    first = {k: np.zeros_like(v) for k, v in mean.items()}
    second = copy.deepcopy(first)
    params = {k: torch.tensor(mean[k][0], dtype=torch.float64, requires_grad=True) for k in KEYS}
    optimizer = torch.optim.Adam(
        list(params.values()), lr=0.001, betas=(0.9, 0.999), eps=1e-8, foreach=False
    )
    rng = np.random.default_rng(913)
    for step in range(1, 21):
        x, y, weight = rng.normal(size=3), rng.normal(size=3), float(rng.uniform(0.3, 2))
        xt, yt = torch.tensor(x), torch.tensor(y)
        optimizer.zero_grad(set_to_none=True)
        output = params["w2"] @ torch.relu(params["w1"] @ xt + params["b1"]) + params["b2"]
        loss = 0.5 * weight * (output - yt).square().sum()
        loss.backward()
        optimizer.step()
        adam_step(
            mean, first, second, x[None], y[None], np.array([weight]), np.array([0.001]), step
        )
        for k in KEYS:
            np.testing.assert_allclose(
                mean[k][0], params[k].detach().numpy(), atol=2e-12, rtol=2e-12
            )
```

</details>

### `test_independent_trajectories_and_checkpoint_prefix` · L158

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('kind', ['adam', 'tagi_diag', 'tagi_full3'])
def test_independent_trajectories_and_checkpoint_prefix(kind):
    x, y, weights = synthetic()
    parameter = 0.001 if kind == "adam" else 0.3
    cases = [dict(method=kind, parameter=parameter, seed=seed) for seed in (17, 29)]
    multiple, metadata = train_block(x, y, weights, "mean3", cases, (1, 2), hidden=6)
    assert len(metadata) == 2
    for i, case in enumerate(cases):
        alone, _ = train_block(x, y, weights, "mean3", [case], (1, 2), hidden=6)
        prefix, _ = train_block(x, y, weights, "mean3", [case], (1,), hidden=6)
        for epoch in (1, 2):
            for key in alone[(0, epoch)]:
                np.testing.assert_allclose(
                    multiple[(i, epoch)][key], alone[(0, epoch)][key], atol=0, rtol=0
                )
        for key in prefix[(0, 1)]:
            np.testing.assert_array_equal(prefix[(0, 1)][key], alone[(0, 1)][key])
```

</details>

### `test_consumer_normalization_and_numeric_capacity` · L177

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('group,expected_bytes', [('mean3', 1855), ('raw36', 10564)])
def test_consumer_normalization_and_numeric_capacity(group, expected_bytes):
    x, y, _ = synthetic()
    prep, xn, yn = prepare(x, y, group)
    d = 3 if group == "mean3" else 36
    mean, _ = initialize(d, 17)
    model = export(prep, mean)
    assert sum(v.nbytes for v in model.values()) == expected_bytes
    call = Predictor(model)
    actual = np.array([call(row) for row in x])
    np.testing.assert_allclose(actual, predict(model, x), atol=1e-12, rtol=1e-12)
    indices = [27, 28, 29] if group == "mean3" else list(range(36))
    wanted_x = ((x[:, indices] - model["x_mean"]) / model["x_std"]).astype(np.float64)
    wanted_y = ((y.astype(np.float32) - model["y_mean"]) / model["y_std"]).astype(np.float64)
    np.testing.assert_array_equal(xn, wanted_x)
    np.testing.assert_array_equal(yn, wanted_y)
    for field, data in (("x", x[:, indices]), ("y", y)):
        np.testing.assert_array_equal(
            model[field + "_mean"], data.astype(np.float64).mean(0).astype(np.float32)
        )
    if group == "mean3":
        changed = x.copy()
        changed[:, :27] = 999
        changed[:, 30:] = -999
        np.testing.assert_array_equal(predict(model, changed), predict(model, x))
    assert call.cached_array_bytes >= expected_bytes
```

</details>

### `test_seed_initialization_and_validation` · L204

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_seed_initialization_and_validation():
    m1, v1 = initialize(3, 17)
    m2, v2 = initialize(3, 17)
    for k in KEYS:
        np.testing.assert_array_equal(m1[k], m2[k])
        np.testing.assert_array_equal(v1[k], v2[k])
        assert np.all(v1[k] > 0)
    x, y, w = synthetic()
    for bad_x, bad_y, group in (
        (x[:, :3], y, "mean3"),
        (x, y[:-1], "raw36"),
        (x, y, "invalid"),
        (x * np.nan, y, "raw36"),
    ):
        with pytest.raises(ValueError):
            prepare(bad_x, bad_y, group)
    with pytest.raises(ValueError):
        train_block(x, y, -w, "raw36", [dict(method="adam", seed=17, parameter=0.001)], (1,))
    prep, _, _ = prepare(x, y, "mean3")
    model = export(prep, m1)
    with pytest.raises(ValueError):
        Predictor({**model, "w1": model["w1"].astype(np.float64)})
    with pytest.raises(ValueError):
        Predictor({**model, "feature_indices": np.array([27, 27, 29], np.uint8)})
    call = Predictor(model)
    with pytest.raises(ValueError):
        call(np.zeros(3))
    with pytest.raises(ValueError):
        call(np.full(36, np.nan))
```

</details>

### `test_candidate_ties_prefer_fewer_epochs_then_declared_order` · L235

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_candidate_ties_prefer_fewer_epochs_then_declared_order():
    candidates = [
        dict(clean=5.0, p90=9.0, epoch=e, parameter_index=i) for e, i in ((64, 0), (4, 2), (4, 1))
    ]
    assert choose(candidates) == candidates[2]
    candidates[0]["clean"] = 4.99
    assert choose(candidates) == candidates[0]
```

</details>

### `test_independent_scalar_training_trace` · L245

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('method', ['adam', 'tagi_diag', 'tagi_full3'])
def test_independent_scalar_training_trace(method):
    x, y, weights = synthetic()
    parameter = 0.001 if method == "adam" else 0.3
    case = dict(method=method, parameter=parameter, seed=29)
    models, records = train_block(x, y, weights, "mean3", [case], (1, 2), hidden=6)
    reference, trace = train_reference(
        x, y, weights, "mean3", method, parameter, 29, (1, 2), hidden=6
    )
    for epoch in (1, 2):
        for k in reference[epoch]:
            np.testing.assert_allclose(
                models[(0, epoch)][k], reference[epoch][k], atol=2e-6, rtol=2e-6
            )
        observed = records[0]["checkpoints"][epoch - 1]
        for k in (
            "floored_variance_updates",
            "negative_variance_updates",
            "order_sha256",
            "examples_seen",
        ):
            assert observed[k] == trace[epoch - 1][k]
```

</details>

### `test_full_width_long_schedule_is_finite_and_counted` · L269

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('method', ['adam', 'tagi_diag', 'tagi_full3'])
def test_full_width_long_schedule_is_finite_and_counted(method):
    x, y, weights = synthetic()
    cases = [dict(method=method, parameter=p, seed=17) for p in PARAMETERS[method]]
    models, records = train_block(x, y, weights, "raw36", cases, (1, 4, 16, 64))
    assert len(models) == 12
    for record in records:
        assert [c["examples_seen"] for c in record["checkpoints"]] == [
            len(x) * e for e in (1, 4, 16, 64)
        ]
        if method == "tagi_full3":
            assert all(c["negative_variance_updates"] == 0 for c in record["checkpoints"])
        if method != "adam":
            assert all(c["minimum_variance"] >= 1e-12 for c in record["checkpoints"])
    for model in models.values():
        assert np.isfinite(predict(model, x)).all()
    if method == "tagi_full3":
        reference, _ = train_reference(
            x, y, weights, "raw36", method, PARAMETERS[method][1], 17, (64,)
        )
        for key in reference[64]:
            np.testing.assert_allclose(
                models[(1, 64)][key], reference[64][key], atol=2e-6, rtol=2e-6
            )
```

</details>
