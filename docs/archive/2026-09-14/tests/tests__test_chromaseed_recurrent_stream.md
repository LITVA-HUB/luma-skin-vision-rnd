# `tests/test_chromaseed_recurrent_stream.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_recurrent_stream.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

CPU-only streaming equivalence and genuine computation omission.

SHA-256 исходника: `f9fc65ad86026951c8022b0854aaa02bb1be1e25169cf4b4997fa20f3df4817c`. Строк: **210**.

## Зависимости

```python
import subprocess
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from chromaseed_architecture_scale import specs
from chromaseed_head_range import Predictor
from chromaseed_recurrent_stream import StreamingPredictor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../tests/test_chromaseed_recurrent_stream.py#L18)

```python
VARIANTS = ("soft_small", "dynamic_small", "soft5m", "dynamic5m")
```

[Строка 19](../../../../tests/test_chromaseed_recurrent_stream.py#L19)

```python
MODES = ("unit", "wide", "linear")
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Observed` | StreamingPredictor | [L52](../../../../tests/test_chromaseed_recurrent_stream.py#L52) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `single_thread` | FunctionDef | См. реализацию | [L23](../../../../tests/test_chromaseed_recurrent_stream.py#L23) |
| `synthetic` | FunctionDef | См. реализацию | [L28](../../../../tests/test_chromaseed_recurrent_stream.py#L28) |
| `Observed` | ClassDef | См. реализацию | [L52](../../../../tests/test_chromaseed_recurrent_stream.py#L52) |
| `test_every_prefix_matches_frozen_consumer_with_nonzero_weights` | FunctionDef | См. реализацию | [L64](../../../../tests/test_chromaseed_recurrent_stream.py#L64) |
| `test_iterator_is_lazy_and_does_not_execute_discarded_passes` | FunctionDef | См. реализацию | [L85](../../../../tests/test_chromaseed_recurrent_stream.py#L85) |
| `test_explicit_threshold_really_stops_and_default_does_not` | FunctionDef | См. реализацию | [L99](../../../../tests/test_chromaseed_recurrent_stream.py#L99) |
| `test_snapshots_and_interleaved_requests_do_not_modify_recurrent_state` | FunctionDef | См. реализацию | [L116](../../../../tests/test_chromaseed_recurrent_stream.py#L116) |
| `test_dynamic_fallback_keeps_exactly_four_tied_nonpositive_tokens` | FunctionDef | См. реализацию | [L136](../../../../tests/test_chromaseed_recurrent_stream.py#L136) |
| `test_invalid_exit_policy_fails_before_any_layer` | FunctionDef | См. реализацию | [L157](../../../../tests/test_chromaseed_recurrent_stream.py#L157) |
| `test_invalid_features_and_payloads_are_rejected` | FunctionDef | См. реализацию | [L165](../../../../tests/test_chromaseed_recurrent_stream.py#L165) |
| `test_standalone_consumer_does_not_import_torch` | FunctionDef | См. реализацию | [L179](../../../../tests/test_chromaseed_recurrent_stream.py#L179) |
| `test_qualification_checks_real_prefix_work_and_no_quality_claim` | FunctionDef | См. реализацию | [L187](../../../../tests/test_chromaseed_recurrent_stream.py#L187) |
| `test_qualification_rejects_an_input_that_changed` | FunctionDef | См. реализацию | [L201](../../../../tests/test_chromaseed_recurrent_stream.py#L201) |

## Все тестовые определения (10)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_every_prefix_matches_frozen_consumer_with_nonzero_weights` · L64

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('mode', MODES)
def test_every_prefix_matches_frozen_consumer_with_nonzero_weights(variant, mode):
    model, x, t = synthetic(variant, mode)
    expected = Predictor(model)(x, t, all_passes=True)
    consumer = StreamingPredictor(model)
    actual = list(consumer.iter_passes(x, t))
    np.testing.assert_array_equal(np.stack([p.prediction for p in actual]), expected)
    assert np.max(np.abs(np.diff(expected, axis=0))) > 1e-3
    assert [p.index for p in actual] == [1, 2, 3, 4]
    assert actual[0].change_l2 is None
    for i in range(1, 4):
        assert actual[i].change_l2 == pytest.approx(np.linalg.norm(expected[i] - expected[i-1]))
    assert all(4 <= p.active_tokens <= 64 for p in actual)
    if variant.startswith("soft"):
        assert all(p.active_tokens == 64 for p in actual)
    for count in (1, 2, 3, 4):
        trace = consumer.predict_trace(x, t, max_passes=count)
        assert trace.passes_executed == count and trace.stop_reason == "max_passes"
        np.testing.assert_array_equal(trace.prediction, expected[count-1])
    np.testing.assert_array_equal(consumer.predict(x, t), expected[-1])
```

</details>

### `test_iterator_is_lazy_and_does_not_execute_discarded_passes` · L85

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_iterator_is_lazy_and_does_not_execute_discarded_passes():
    model, x, t = synthetic()
    consumer = Observed(model)
    stream = consumer.iter_passes(x, t)
    assert consumer.calls == []
    next(stream)
    first = ["token1", "token2", "context", "key", "query", "update1", "update2", "head"]
    assert consumer.calls == first
    next(stream)
    assert consumer.calls == first + ["query", "update1", "update2", "head"]
    stream.close()
    assert len(consumer.calls) == 12
```

</details>

### `test_explicit_threshold_really_stops_and_default_does_not` · L99

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_explicit_threshold_really_stops_and_default_does_not():
    model, x, t = synthetic()
    model["exit_threshold"] = np.asarray(1e6)  # An old payload setting is not adopted.
    consumer = Observed(model)
    expected = Predictor(model)(x, t, all_passes=True)
    trace = consumer.predict_trace(x, t, exit_threshold=1e6)
    assert trace.passes_executed == 2 and trace.stop_reason == "delta_threshold"
    np.testing.assert_array_equal(trace.prediction, expected[1])
    counts = Counter(consumer.calls)
    assert counts == Counter(token1=1, token2=1, context=1, key=1, query=2, update1=2, update2=2, head=2)
    consumer.calls.clear()
    assert consumer.predict_trace(x, t).passes_executed == 4
    assert Counter(consumer.calls)["head"] == 4
    assert consumer.predict_trace(x, t, exit_threshold=1e6, min_passes=3).passes_executed == 3
    assert consumer.predict_trace(x, t, exit_threshold=1e-12).passes_executed == 4
```

</details>

### `test_snapshots_and_interleaved_requests_do_not_modify_recurrent_state` · L116

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_snapshots_and_interleaved_requests_do_not_modify_recurrent_state():
    model, x, t = synthetic()
    consumer = StreamingPredictor(model)
    a = consumer.iter_passes(x, t)
    b = consumer.iter_passes(-x, -t)
    first = next(a)
    saved = first.prediction.copy()
    next(b)
    second = next(a)
    np.testing.assert_array_equal(first.prediction, saved)
    # Public output storage never aliases the private recurrent prediction.
    second.prediction.setflags(write=True)
    second.prediction[:] = 1e20
    final = list(a)[-1].prediction
    np.testing.assert_array_equal(final, consumer.predict(x, t))
    np.testing.assert_array_equal(list(b)[-1].prediction, consumer.predict(-x, -t))
    model["theta"][:] = 0
    np.testing.assert_array_equal(consumer.predict(x, t), final)
```

</details>

### `test_dynamic_fallback_keeps_exactly_four_tied_nonpositive_tokens` · L136

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_dynamic_fallback_keeps_exactly_four_tied_nonpositive_tokens():
    model, x, t = synthetic()
    offset = 0
    for name, incoming, outgoing in specs("dynamic_small"):
        size = (incoming + (name != "key")) * outgoing
        if name == "key":
            model["theta"][offset:offset+size] = 0
        offset += size
    consumer = StreamingPredictor(model)
    passes = list(consumer.iter_passes(x, t))
    assert all(p.active_tokens == 4 for p in passes)
    np.testing.assert_array_equal(np.stack([p.prediction for p in passes]),
                                  Predictor(model)(x, t, all_passes=True))
```

</details>

### `test_invalid_exit_policy_fails_before_any_layer` · L157

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('field,value', [('max_passes', 0), ('max_passes', 5), ('max_passes', True), ('max_passes', 2.5), ('min_passes', 1), ('min_passes', 5), ('min_passes', True), ('exit_threshold', -1), ('exit_threshold', float('nan')), ('exit_threshold', float('inf')), ('exit_threshold', True)])
def test_invalid_exit_policy_fails_before_any_layer(field, value):
    model, x, t = synthetic()
    consumer = Observed(model)
    with pytest.raises(ValueError):
        consumer.predict_trace(x, t, **{field: value})
    assert not consumer.calls
```

</details>

### `test_invalid_features_and_payloads_are_rejected` · L165

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_features_and_payloads_are_rejected():
    model, x, t = synthetic()
    consumer = Observed(model)
    for xx, tt in ((x[None], t), (x, t[None]), (x[:-1], t), (x, t[:-1]), (x*np.nan, t)):
        with pytest.raises(ValueError):
            consumer.predict(xx, tt)
    assert not consumer.calls
    for key, value in (("variant", "patch5m"), ("head_mode", "unknown"),
                       ("theta", model["theta"][:-1]), ("base_x_std", np.zeros(36)),
                       ("t_std", np.ones((1, 18))), ("base_y_mean", [1, 2, np.inf])):
        with pytest.raises(ValueError):
            StreamingPredictor({**model, key: value})
```

</details>

### `test_standalone_consumer_does_not_import_torch` · L179

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_standalone_consumer_does_not_import_torch():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    code = (f"import sys; sys.path.insert(0, {scripts!r}); "
            "import chromaseed_recurrent_stream; assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

</details>

### `test_qualification_checks_real_prefix_work_and_no_quality_claim` · L187

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_qualification_checks_real_prefix_work_and_no_quality_claim():
    from chromaseed_recurrent_stream_check import qualify_case

    model, x, t = synthetic()
    result = qualify_case(model, x, t)
    assert result["all_passes_bitwise_equal"]
    assert result["max_abs_lab_drift"] == 0.0
    assert [p["linear_macs"] for p in result["prefixes"]] == [135336, 144912, 154488, 164064]
    assert result["forced_exit"]["passes_executed"] == 2
    assert result["forced_exit"]["linear_macs"] == 144912
    assert result["default_passes"] == 4
    assert result["quality_metric"] is None and result["latency_seconds"] is None
```

</details>

### `test_qualification_rejects_an_input_that_changed` · L201

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_qualification_rejects_an_input_that_changed(tmp_path):
    from chromaseed_recurrent_stream_check import binding, check_bindings

    source = tmp_path / "source.txt"
    source.write_text("before", encoding="utf-8")
    records = [binding(source)]
    check_bindings(records)
    source.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        check_bindings(records)
```

</details>
