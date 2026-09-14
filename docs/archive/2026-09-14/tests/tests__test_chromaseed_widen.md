# `tests/test_chromaseed_widen.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_widen.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Widening must preserve the old function while enabling added neurons to learn.

SHA-256 исходника: `bd4454a97282201eefd9756a20ea46be1c92ceb60763274c33fabcec8b1c7ce6`. Строк: **108**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_patch8_fit import token_normalizers
from chromaseed_widen import SPECS, Bank, Predictor, capacity, fit, predict, rate_factors
from test_chromaseed_patch8 import fixture
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_widening_preserves_function_and_adds_real_capacity` | FunctionDef | См. реализацию | [L21](../../../../tests/test_chromaseed_widen.py#L21) |
| `test_added_units_and_patch_branch_learn_and_are_permutation_invariant` | FunctionDef | См. реализацию | [L39](../../../../tests/test_chromaseed_widen.py#L39) |
| `test_zero_variance_patches_have_finite_gradients` | FunctionDef | См. реализацию | [L51](../../../../tests/test_chromaseed_widen.py#L51) |
| `test_cpu_short_replay_keeps_original_horizon_and_rate_baseline` | FunctionDef | См. реализацию | [L62](../../../../tests/test_chromaseed_widen.py#L62) |
| `test_large_cuda_graph_replay_and_export_forward_agree` | FunctionDef | См. реализацию | [L76](../../../../tests/test_chromaseed_widen.py#L76) |
| `test_consumer_validates_payload_and_input` | FunctionDef | См. реализацию | [L98](../../../../tests/test_chromaseed_widen.py#L98) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_widening_preserves_function_and_adds_real_capacity` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('variant,count', [('tiny', 1179), ('m31', 30915), ('m61', 60611), ('m111', 110979), ('m832', 832259)])
def test_widening_preserves_function_and_adds_real_capacity(variant, count):
    x, t, _, warm = fixture(5)
    net = Bank(warm, variant)
    prep = token_normalizers(t)
    assert net.theta.shape == (6, count)
    for slot in range(6):
        m = net.export(slot, warm, prep)
        assert capacity(m)["parameters"] == count
        np.testing.assert_allclose(
            predict(m, x, t), base_predict(warm[slot // 2], x), rtol=0, atol=2e-8
        )
        assert np.count_nonzero(m["v0"][16:]) == 0
        assert np.count_nonzero(m["g"]) == 0
        changed = m["w0"].copy()
        m["w0"].fill(99)
        np.testing.assert_array_equal(net.export(slot, warm, prep)["w0"], changed)
```

</details>

### `test_added_units_and_patch_branch_learn_and_are_permutation_invariant` · L39

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_added_units_and_patch_branch_learn_and_are_permutation_invariant():
    x, t, y, warm = fixture()
    result, info = fit(x, t, y, np.ones(len(x)), warm, "m31", 4, (0, 4))
    m = result[4][0]
    assert np.any(m["v0"][16:] != 0) and np.any(m["g"] != 0)
    assert np.any(m["u"] != result[0][0]["u"])
    np.testing.assert_allclose(
        predict(m, x, t), predict(m, x, t[:, ::-1].copy()), rtol=0, atol=2e-8
    )
    assert info["trajectory_count"] == 6 and info["schedule_horizon"] == 8192
```

</details>

### `test_zero_variance_patches_have_finite_gradients` · L51

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_zero_variance_patches_have_finite_gradients():
    x, _, _, warm = fixture()
    net = Bank(warm, "m31")
    with torch.no_grad():
        net.parts()["g"].fill_(0.001)
    net(
        torch.tensor(np.repeat(x[:2][None], 6, axis=0)), torch.zeros(6, 2, 64, 18)
    ).square().sum().backward()
    assert torch.isfinite(net.theta.grad).all()
```

</details>

### `test_cpu_short_replay_keeps_original_horizon_and_rate_baseline` · L62

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_cpu_short_replay_keeps_original_horizon_and_rate_baseline():
    x, t, y, warm = fixture()
    a, ia = fit(x, t, y, np.ones(len(x)), warm, "tiny", 4, (0, 4))
    b, ib = fit(x, t, y, np.ones(len(x)), warm, "tiny", 8, (0, 4, 8))
    for i in range(6):
        for k in a[4][i]:
            np.testing.assert_array_equal(a[4][i][k], b[4][i][k])
    np.testing.assert_array_equal(rate_factors(4), rate_factors(8)[:4])
    expected = np.tile(np.array([0.0001, 0.001], np.float32), 3) * rate_factors(4)[-1]
    np.testing.assert_array_equal(np.array(ia["final_learning_rates"], np.float32), expected)
    assert ib["sampling_sha256"] != ia["sampling_sha256"]
```

</details>

### `test_large_cuda_graph_replay_and_export_forward_agree` · L76

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA absent')
def test_large_cuda_graph_replay_and_export_forward_agree():
    x, t, y, warm = fixture(6)
    a, _ = fit(x, t, y, np.ones(len(x)), warm, "m832", 4, (0, 4), "cuda", "cuda_graph")
    b, _ = fit(x, t, y, np.ones(len(x)), warm, "m832", 8, (0, 4, 8), "cuda", "cuda_graph")
    for slot in range(6):
        for k in a[4][slot]:
            np.testing.assert_array_equal(a[4][slot][k], b[4][slot][k])
    net = Bank(warm, "m832").cuda()
    net.load_models(a[4])
    prep = token_normalizers(t)
    xx = torch.tensor(
        np.repeat(((x - warm[0]["x_mean"]) / warm[0]["x_std"])[None], 6, axis=0), device="cuda"
    )
    tt = torch.tensor(
        np.repeat(((t - prep["t_mean"]) / prep["t_std"])[None], 6, axis=0), device="cuda"
    )
    with torch.no_grad():
        output = net(xx, tt).cpu().numpy() * warm[0]["y_std"] + warm[0]["y_mean"]
    for slot in range(6):
        np.testing.assert_allclose(output[slot], predict(a[4][slot], x, t), rtol=0, atol=0.002)
```

</details>

### `test_consumer_validates_payload_and_input` · L98

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_consumer_validates_payload_and_input():
    x, t, _, warm = fixture()
    m = Bank(warm, "m111").export(0, warm, token_normalizers(t))
    assert SPECS["m111"] == (128, 256)
    consumer = Predictor(m)
    np.testing.assert_allclose(consumer(x[0], t[0]), predict(m, x[:1], t[:1])[0], rtol=0, atol=2e-8)
    with pytest.raises(ValueError):
        consumer(x[0], t[:1])
    m["t_std"].fill(0)
    with pytest.raises(ValueError):
        Predictor(m)
```

</details>
