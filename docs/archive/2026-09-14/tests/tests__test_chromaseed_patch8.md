# `tests/test_chromaseed_patch8.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_patch8.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

P8 behavior: exact warm mapping, new local information and reproducible training.

SHA-256 исходника: `62f51d9f1d72285bb2920a29596c0b0f45a2f3d85f1035d5cc24f9bba028df4d`. Строк: **188**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from chromaseed_neural_prefix_numpy import predict as np_predict
from chromaseed_patch8_fit import Bank, fit, rate_factors, token_normalizers
from chromaseed_patch8_numpy import Predictor, choose, predict, transform_tokens
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fixture` | FunctionDef | См. реализацию | [L16](../../../../tests/test_chromaseed_patch8.py#L16) |
| `test_initial_mapping_sizes_and_fresh_export` | FunctionDef | См. реализацию | [L43](../../../../tests/test_chromaseed_patch8.py#L43) |
| `test_local_inputs_affect_activated_branch_and_permutation_invariance` | FunctionDef | См. реализацию | [L61](../../../../tests/test_chromaseed_patch8.py#L61) |
| `independent_tokens` | FunctionDef | См. реализацию | [L69](../../../../tests/test_chromaseed_patch8.py#L69) |
| `test_joint_token_affine_law_on_generated_continuous_rgb` | FunctionDef | См. реализацию | [L77](../../../../tests/test_chromaseed_patch8.py#L77) |
| `test_token_normalizers_use_given_fit_rows_only` | FunctionDef | См. реализацию | [L87](../../../../tests/test_chromaseed_patch8.py#L87) |
| `test_constant_token_backward_is_finite_and_stats_branch_has_zero_gradient` | FunctionDef | См. реализацию | [L95](../../../../tests/test_chromaseed_patch8.py#L95) |
| `test_fit_wakes_patch_and_keeps_stats_g_zero_cpu` | FunctionDef | См. реализацию | [L108](../../../../tests/test_chromaseed_patch8.py#L108) |
| `test_original_horizon_rate_prefix_and_cpu_no_cumulative_decay` | FunctionDef | См. реализацию | [L117](../../../../tests/test_chromaseed_patch8.py#L117) |
| `test_fixed_shape_graph_replay_and_eager_parity` | FunctionDef | См. реализацию | [L130](../../../../tests/test_chromaseed_patch8.py#L130) |
| `test_cuda_forward_matches_actual_numpy_local_consumer` | FunctionDef | См. реализацию | [L146](../../../../tests/test_chromaseed_patch8.py#L146) |
| `test_selector_keeps_smaller_baseline_on_tie_but_prefers_quality` | FunctionDef | См. реализацию | [L172](../../../../tests/test_chromaseed_patch8.py#L172) |
| `test_consumer_rejects_missing_branch_or_bad_normalizers` | FunctionDef | См. реализацию | [L180](../../../../tests/test_chromaseed_patch8.py#L180) |

## Все тестовые определения (11)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_initial_mapping_sizes_and_fresh_export` · L43

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_initial_mapping_sizes_and_fresh_export():
    x, t, y, warm = fixture()
    net = Bank(warm)
    prep = token_normalizers(t)
    assert net.theta.shape == (18, 1179)
    for slot in range(18):
        m = net.export(slot, warm, prep)
        np.testing.assert_array_equal(predict(m, x, t), np_predict(warm[slot // 6], x))
        patch = (slot % 6) // 3 == 1
        assert sum(v.nbytes for v in m.values() if v.dtype.kind in "biufc") == (
            5174 if patch else 2886
        )
        assert Predictor(m).cached_array_bytes == (9888 if patch else 5456)
    a = net.export(3, warm, prep)
    a["w0"].fill(100)
    assert not np.any(net.export(3, warm, prep)["w0"] == 100)
```

</details>

### `test_local_inputs_affect_activated_branch_and_permutation_invariance` · L61

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_local_inputs_affect_activated_branch_and_permutation_invariance():
    x, t, _, warm = fixture()
    m = Bank(warm).export(3, warm, token_normalizers(t))
    m["g"].fill(0.03)
    assert np.max(abs(predict(m, x, t) - predict(m, x, np.flip(t, axis=1).copy()))) < 2e-8
    assert np.max(abs(predict(m, x, t) - predict(m, x, np.zeros_like(t)))) > 0.01
```

</details>

### `test_joint_token_affine_law_on_generated_continuous_rgb` · L77

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_joint_token_affine_law_on_generated_continuous_rgb():
    rgb = np.random.default_rng(2).uniform(size=(128, 128, 3))
    tokens = independent_tokens(rgb).astype(np.float32)
    for dose in (0.0, 4 / 255, 64 / 255):
        a = np.array([0.1, 0.6, 0.9])
        expected = independent_tokens(rgb * (1 - dose) + dose * a).astype(np.float32)
        np.testing.assert_allclose(transform_tokens(tokens, dose, a), expected, rtol=0, atol=6e-8)
    np.testing.assert_array_equal(transform_tokens(tokens, 0, np.ones(3)), tokens)
```

</details>

### `test_token_normalizers_use_given_fit_rows_only` · L87

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_token_normalizers_use_given_fit_rows_only():
    _, t, _, _ = fixture()
    p = token_normalizers(t[:4])
    np.testing.assert_array_equal(p["t_mean"], t[:4].astype(float).mean((0, 1)).astype(np.float32))
    assert not np.array_equal(p["t_mean"], token_normalizers(t)["t_mean"])
    assert np.all(token_normalizers(np.zeros_like(t))["t_std"] == np.float32(1e-6))
```

</details>

### `test_constant_token_backward_is_finite_and_stats_branch_has_zero_gradient` · L95

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_constant_token_backward_is_finite_and_stats_branch_has_zero_gradient():
    x, t, _, warm = fixture()
    net = Bank(warm)
    with torch.no_grad():
        net.theta[:, 795:] = 0.01
    out = net(torch.tensor(np.repeat(x[:2][None], 18, axis=0)), torch.zeros(18, 2, 64, 18))
    out.square().sum().backward()
    assert torch.isfinite(net.theta.grad).all()
    for slot in range(18):
        if slot % 6 < 3:
            assert torch.count_nonzero(net.theta.grad[slot, 643:]) == 0
```

</details>

### `test_fit_wakes_patch_and_keeps_stats_g_zero_cpu` · L108

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fit_wakes_patch_and_keeps_stats_g_zero_cpu():
    x, t, y, warm = fixture()
    result, info = fit(x, t, y, np.ones(len(x)), warm, 4, (0, 4))
    assert info["trajectory_count"] == 18
    assert np.any(result[4][3]["g"] != 0)
    assert "g" not in result[4][0]
    assert np.isfinite(predict(result[4][3], x, t)).all()
```

</details>

### `test_original_horizon_rate_prefix_and_cpu_no_cumulative_decay` · L117

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_original_horizon_rate_prefix_and_cpu_no_cumulative_decay():
    np.testing.assert_array_equal(rate_factors(8), rate_factors(32768)[:8])
    assert rate_factors(32768)[-1] == np.float32(0.1)
    x, t, y, warm = fixture(6)
    _, info = fit(x, t, y, np.ones(len(x)), warm, 16, (0, 16))
    expected = (
        np.array([r for _ in range(6) for r in (0.0001, 0.0003, 0.001)], np.float32)
        * rate_factors(16)[-1]
    )
    np.testing.assert_array_equal(np.array(info["final_learning_rates"], np.float32), expected)
```

</details>

### `test_fixed_shape_graph_replay_and_eager_parity` · L130

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_fixed_shape_graph_replay_and_eager_parity():
    x, t, y, warm = fixture(6)
    a, _ = fit(x, t, y, np.ones(len(x)), warm, 8, (0, 8), "cuda", "cuda_graph")
    b, _ = fit(x, t, y, np.ones(len(x)), warm, 16, (0, 8, 16), "cuda", "cuda_graph")
    c, _ = fit(x, t, y, np.ones(len(x)), warm, 8, (0, 8), "cuda", "eager")
    for i in range(18):
        for k in a[8][i]:
            np.testing.assert_array_equal(a[8][i][k], b[8][i][k])
            if a[8][i][k].dtype.kind == "f":
                np.testing.assert_allclose(a[8][i][k], c[8][i][k], rtol=0, atol=2e-6)
        np.testing.assert_allclose(
            predict(a[8][i], x, t), predict(c[8][i], x, t), rtol=0, atol=0.002
        )
```

</details>

### `test_cuda_forward_matches_actual_numpy_local_consumer` · L146

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_forward_matches_actual_numpy_local_consumer():
    x, t, _, warm = fixture()
    net = Bank(warm).cuda()
    with torch.no_grad():
        net.theta[:, 795:] = 0.02
        net.theta[:3, 795:] = 0
        net.theta[6:9, 795:] = 0
        net.theta[12:15, 795:] = 0
    prep = token_normalizers(t)
    xn = (x - warm[0]["x_mean"]) / warm[0]["x_std"]
    tn = (t - prep["t_mean"]) / prep["t_std"]
    with torch.no_grad():
        actual = (
            net(
                torch.tensor(np.repeat(xn[None], 18, axis=0), device="cuda"),
                torch.tensor(np.repeat(tn[None], 18, axis=0), device="cuda"),
            )
            .cpu()
            .numpy()
        )
    actual = actual * warm[0]["y_std"] + warm[0]["y_mean"]
    for slot in range(18):
        expected = predict(net.export(slot, warm, prep), x, t)
        np.testing.assert_allclose(actual[slot], expected, rtol=0, atol=0.002)
```

</details>

### `test_selector_keeps_smaller_baseline_on_tie_but_prefers_quality` · L172

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selector_keeps_smaller_baseline_on_tie_but_prefers_quality():
    baseline = dict(clean=5.0, p90=9.0, numeric_bytes=2886, step=0, lr=None, arm="stats")
    patch = dict(clean=5.0, p90=9.0, numeric_bytes=5174, step=512, lr=0.0001, arm="patch8")
    assert choose([patch, baseline]) is baseline
    patch["clean"] = 4.9
    assert choose([patch, baseline]) is patch
```

</details>

### `test_consumer_rejects_missing_branch_or_bad_normalizers` · L180

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_consumer_rejects_missing_branch_or_bad_normalizers():
    _, t, _, warm = fixture()
    m = Bank(warm).export(3, warm, token_normalizers(t))
    bad = {k: v for k, v in m.items() if k != "g"}
    with pytest.raises(ValueError):
        Predictor(bad)
    m["t_std"][0] = 0
    with pytest.raises(ValueError):
        Predictor(m)
```

</details>
