# `tests/test_chromaseed_neural_prefix.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_neural_prefix.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Behavioral tests for exact prefixes, fixed schedules and conditional selection.

SHA-256 исходника: `8b497ed429e1a0c7a2b784d5d2010f7d1449acc00e373a611190e5d7e7231065`. Строк: **136**.

## Зависимости

```python
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `mod` | FunctionDef | См. реализацию | [L14](../../../../tests/test_chromaseed_neural_prefix.py#L14) |
| `parent` | FunctionDef | См. реализацию | [L19](../../../../tests/test_chromaseed_neural_prefix.py#L19) |
| `test_every_prefix_matches_original_clean_head` | FunctionDef | См. реализацию | [L39](../../../../tests/test_chromaseed_neural_prefix.py#L39) |
| `test_prefix_two_keeps_original_four_stage_schedule_and_returns_clean` | FunctionDef | См. реализацию | [L56](../../../../tests/test_chromaseed_neural_prefix.py#L56) |
| `test_blind_head_collapses_and_ignores_earlier_heads_and_state_weights` | FunctionDef | См. реализацию | [L72](../../../../tests/test_chromaseed_neural_prefix.py#L72) |
| `test_payload_size_counts_only_retained_weights_and_metadata` | FunctionDef | См. реализацию | [L85](../../../../tests/test_chromaseed_neural_prefix.py#L85) |
| `test_invalid_export_prefix_rejected` | FunctionDef | См. реализацию | [L95](../../../../tests/test_chromaseed_neural_prefix.py#L95) |
| `test_single_consumer_rejects_invalid_input` | FunctionDef | См. реализацию | [L101](../../../../tests/test_chromaseed_neural_prefix.py#L101) |
| `test_malformed_payload_rejected` | FunctionDef | См. реализацию | [L116](../../../../tests/test_chromaseed_neural_prefix.py#L116) |
| `test_compact_policy_obeys_error_tolerance_and_tie_order` | FunctionDef | См. реализацию | [L124](../../../../tests/test_chromaseed_neural_prefix.py#L124) |

## Все тестовые определения (8)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_every_prefix_matches_original_clean_head` · L39

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family', ['plain', 'local2', 'local4', 'blind4', 'e2e4'])
def test_every_prefix_matches_original_clean_head(family):
    m = mod()
    from chromaseed_local_denoise_audit import direct

    p = parent(family)
    x = np.random.default_rng(34).normal(size=(11, 36)).astype(np.float32)
    expected = direct(p, x)
    for j in range(1, expected.shape[1] + 1):
        export = m.export_prefix(p, j)
        np.testing.assert_allclose(m.predict(export, x), expected[:, j - 1], rtol=0, atol=2e-8)
        single = m.Predictor(export)
        np.testing.assert_allclose(
            np.stack([single(row) for row in x]), expected[:, j - 1], rtol=0, atol=2e-8
        )
        assert int(export["original_k"]) == len(p["theta"])
```

</details>

### `test_prefix_two_keeps_original_four_stage_schedule_and_returns_clean` · L56

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_prefix_two_keeps_original_four_stage_schedule_and_returns_clean():
    m = mod()
    p = parent("local4")
    p["theta"][:] = 0
    p["y_mean"][:] = 0
    p["theta"][0, -3:] = 1
    # Block2 copies its three nonnegative state coordinates through three ReLU units.
    w = p["theta"][1, : 39 * 16].reshape(39, 16)
    w[36:39, :3] = np.eye(3)
    v = p["theta"][1, 40 * 16 : -3].reshape(16, 3)
    v[:3] = np.eye(3)
    result = m.Predictor(m.export_prefix(p, 2))(np.zeros(36, np.float32))
    np.testing.assert_allclose(result, np.full(3, np.sin(np.pi / 8)), rtol=0, atol=1e-12)
    assert not np.allclose(result, np.sin(np.pi / 4))
```

</details>

### `test_blind_head_collapses_and_ignores_earlier_heads_and_state_weights` · L72

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_blind_head_collapses_and_ignores_earlier_heads_and_state_weights():
    m = mod()
    p = parent("blind4")
    out = m.export_prefix(p, 3)
    assert len([key for key in out if key.startswith("w")]) == 1
    assert out["w0"].shape == (36, 16)
    p["theta"][:2] += 999
    p["theta"][2, : 39 * 16].reshape(39, 16)[36:] += 999
    changed = m.export_prefix(p, 3)
    for key in out:
        np.testing.assert_array_equal(out[key], changed[key])
```

</details>

### `test_payload_size_counts_only_retained_weights_and_metadata` · L85

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_payload_size_counts_only_retained_weights_and_metadata():
    m = mod()
    first = m.export_prefix(parent("e2e4"), 1)
    full = m.export_prefix(parent("e2e4"), 4)
    assert m.capacity(first) == dict(parameters=643, numeric_bytes=2886, executed_blocks=1)
    assert m.capacity(full) == dict(parameters=2716, numeric_bytes=11178, executed_blocks=4)
    assert first["w0"].dtype == np.float32
```

</details>

### `test_invalid_export_prefix_rejected` · L95

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('bad', [0, 5, 1.5, True])
def test_invalid_export_prefix_rejected(bad):
    with pytest.raises(ValueError):
        mod().export_prefix(parent("local4"), bad)
```

</details>

### `test_single_consumer_rejects_invalid_input` · L101

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('bad', [np.zeros((1, 36)), np.zeros(35), np.full(36, np.nan)])
def test_single_consumer_rejects_invalid_input(bad):
    m = mod()
    with pytest.raises(ValueError):
        m.Predictor(m.export_prefix(parent("local4"), 1))(bad)
```

</details>

### `test_malformed_payload_rejected` · L116

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('key,value', [('original_k', np.array(2, np.uint8)), ('prefix', np.array(1.5)), ('w0', np.zeros((39, 16), np.float32)), ('y_std', np.zeros(3, np.float32))])
def test_malformed_payload_rejected(key, value):
    m = mod()
    p = m.export_prefix(parent("local4"), 1)
    p[key] = value
    with pytest.raises(ValueError):
        m.Predictor(p)
```

</details>

### `test_compact_policy_obeys_error_tolerance_and_tie_order` · L124

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_compact_policy_obeys_error_tolerance_and_tie_order():
    m = mod()
    candidates = [
        dict(prefix=1, clean=5.11, p90=8.0, numeric_bytes=2886, executed_blocks=1),
        dict(prefix=2, clean=5.09, p90=8.0, numeric_bytes=5650, executed_blocks=2),
        dict(prefix=3, clean=5.0, p90=9.0, numeric_bytes=8414, executed_blocks=3),
        dict(prefix=4, clean=5.0, p90=8.0, numeric_bytes=11178, executed_blocks=4),
    ]
    chosen = m.choose(candidates)
    assert chosen["quality"]["prefix"] == 4
    assert chosen["compact"]["prefix"] == 2
    candidates[0]["clean"] = 5.1
    assert m.choose(candidates)["compact"]["prefix"] == 1
```

</details>
