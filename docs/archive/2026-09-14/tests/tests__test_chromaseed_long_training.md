# `tests/test_chromaseed_long_training.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_long_training.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

LT synthetic variation and fixed-recipe continuation behavioral checks.

SHA-256 исходника: `4b7e7f489cae80b4093f3fa7e77ecaf7efc31ec4da940b33e1aa87e2944916f8`. Строк: **143**.

## Зависимости

```python
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `mod` | FunctionDef | См. реализацию | [L15](../../../../tests/test_chromaseed_long_training.py#L15) |
| `fixture` | FunctionDef | См. реализацию | [L20](../../../../tests/test_chromaseed_long_training.py#L20) |
| `test_pool_preserves_original_prefix_and_moments` | FunctionDef | См. реализацию | [L40](../../../../tests/test_chromaseed_long_training.py#L40) |
| `test_variant_choice_keeps_clean_half_and_uniform_variant_mass` | FunctionDef | См. реализацию | [L65](../../../../tests/test_chromaseed_long_training.py#L65) |
| `test_sampling_and_learning_rate_prefixes_preserve_original_horizon` | FunctionDef | См. реализацию | [L77](../../../../tests/test_chromaseed_long_training.py#L77) |
| `test_pack_export_preserves_warm_start_without_training` | FunctionDef | См. реализацию | [L86](../../../../tests/test_chromaseed_long_training.py#L86) |
| `test_cpu_continuation_saves_exact_baseline_and_valid_changed_weights` | FunctionDef | См. реализацию | [L101](../../../../tests/test_chromaseed_long_training.py#L101) |
| `test_cpu_rate_schedule_uses_fixed_base_not_cumulative_decay` | FunctionDef | См. реализацию | [L111](../../../../tests/test_chromaseed_long_training.py#L111) |
| `test_selection_can_retain_baseline_and_prefers_smaller_mode_only_on_ties` | FunctionDef | См. реализацию | [L119](../../../../tests/test_chromaseed_long_training.py#L119) |
| `test_fixed_bank_graph_reset_and_short_replay_are_exact` | FunctionDef | См. реализацию | [L131](../../../../tests/test_chromaseed_long_training.py#L131) |

## Все тестовые определения (8)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_pool_preserves_original_prefix_and_moments` · L40

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pool_preserves_original_prefix_and_moments():
    m = mod()
    x, _, rows, _ = fixture()
    pool = m.make_pool(x, rows)
    assert pool.shape == (18, 257, 36)
    np.testing.assert_array_equal(pool[:, 0], x)
    from chromaseed_gate_stability import basic_legal

    assert basic_legal(pool.reshape(-1, 36)).all()
    repeat = m.make_pool(x[[3, 0]], rows[[3, 0]])
    np.testing.assert_array_equal(repeat, pool[[3, 0]])
    draws = np.random.default_rng(880003 + 1009 * 3).random((256, 4))
    t = draws[:, 3] * 4 / 255
    np.testing.assert_allclose(
        pool[3, 1:, 27:30],
        (1 - t[:, None]) * x[3, 27:30] + t[:, None] * draws[:, :3],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        pool[3, 1:, 30:33], (1 - t[:, None]) * x[3, 30:33], rtol=0, atol=1e-7
    )
    np.testing.assert_array_equal(pool[3, 1:17], repeat[0, 1:17])
```

</details>

### `test_variant_choice_keeps_clean_half_and_uniform_variant_mass` · L65

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_variant_choice_keeps_clean_half_and_uniform_variant_mass():
    m = mod()
    u = torch.tensor([[0.0, 0.249, 0.499, 0.5, 0.625, 0.75, 0.875, 0.999]])
    for count in (0, 16, 256):
        result = m.variant_indices(u, torch.tensor([[count]])).numpy()[0]
        if count == 0:
            assert not result.any()
        else:
            np.testing.assert_array_equal(result[:3], [0, 0, 0])
            assert result[3] == 1 and result[-1] <= count and len(set(result[3:])) == 5
```

</details>

### `test_sampling_and_learning_rate_prefixes_preserve_original_horizon` · L77

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_sampling_and_learning_rate_prefixes_preserve_original_horizon():
    m = mod()
    a = m.uniforms((17, 29, 43), 7)
    b = m.uniforms((17, 29, 43), 31)
    np.testing.assert_array_equal(a, b[:, :7])
    np.testing.assert_array_equal(m.lr_factors(7), m.lr_factors(31)[:7])
    assert m.lr_factors(131072)[0] == 1 and abs(m.lr_factors(131072)[-1] - 0.1) < 1e-7
```

</details>

### `test_pack_export_preserves_warm_start_without_training` · L86

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pack_export_preserves_warm_start_without_training():
    m = mod()
    x, _, _, warm = fixture()
    net = m.HeadBank(warm)
    from chromaseed_neural_prefix_numpy import predict

    assert net.theta.shape == (18, 643)
    for i, slot in enumerate(m.SLOTS):
        actual = net.export(i, warm)
        expected = warm[(17, 29, 43).index(slot["seed"])]
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        np.testing.assert_array_equal(predict(actual, x), predict(expected, x))
```

</details>

### `test_cpu_continuation_saves_exact_baseline_and_valid_changed_weights` · L101

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_cpu_continuation_saves_exact_baseline_and_valid_changed_weights():
    m = mod()
    x, y, rows, warm = fixture()
    models, info = m.fit(x, y, np.ones(len(x)), rows, warm, 3, (0, 1, 3))
    assert info["trajectory_count"] == 18 and info["original_rows"] == len(x)
    for key in warm[0]:
        np.testing.assert_array_equal(models[0][0][key], warm[0][key])
    assert not np.array_equal(models[3][0]["w0"], warm[0]["w0"])
```

</details>

### `test_cpu_rate_schedule_uses_fixed_base_not_cumulative_decay` · L111

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_cpu_rate_schedule_uses_fixed_base_not_cumulative_decay():
    m = mod()
    x, y, rows, warm = fixture()
    _, info = m.fit(x, y, np.ones(len(x)), rows, warm, 128, (0, 128))
    expected = np.array([s["lr"] for s in m.SLOTS], np.float32) * m.lr_factors(128)[-1]
    np.testing.assert_array_equal(np.array(info["final_learning_rates"], np.float32), expected)
```

</details>

### `test_selection_can_retain_baseline_and_prefers_smaller_mode_only_on_ties` · L119

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_can_retain_baseline_and_prefers_smaller_mode_only_on_ties():
    from chromaseed_long_training_run import choose

    baseline = dict(clean=5.0, p90=8.0, step=0, variants=0, lr=None)
    same = dict(baseline, variants=256)
    worse = dict(clean=5.1, p90=8.0, step=131072, variants=256, lr=0.0003)
    assert choose([worse, same, baseline], True) == baseline
    better = dict(worse, clean=4.9)
    assert choose([baseline, better], True) == better
```

</details>

### `test_fixed_bank_graph_reset_and_short_replay_are_exact` · L131

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_fixed_bank_graph_reset_and_short_replay_are_exact():
    m = mod()
    x, y, rows, warm = fixture()
    full, _ = m.fit(x, y, np.ones(len(x)), rows, warm, 12, (0, 4, 12), "cuda", "cuda_graph")
    short, _ = m.fit(x, y, np.ones(len(x)), rows, warm, 4, (0, 4), "cuda", "cuda_graph")
    eager, _ = m.fit(x, y, np.ones(len(x)), rows, warm, 4, (0, 4), "cuda", "eager")
    for i in range(18):
        for key in full[4][i]:
            np.testing.assert_array_equal(full[4][i][key], short[4][i][key])
            if full[4][i][key].dtype.kind == "f":
                np.testing.assert_allclose(full[4][i][key], eager[4][i][key], rtol=2e-6, atol=2e-6)
            else:
                np.testing.assert_array_equal(full[4][i][key], eager[4][i][key])
```

</details>
