# `tests/test_chromaseed_refine_precision.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_refine_precision.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `7c1f7389a0569d06163ec5e72d0dbd89f31e91bdfccb57dfffabcb8b69453823`. Строк: **29**.

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
| `test_storage_round_trip_preserves_biases_and_bounds_channel_error` | FunctionDef | См. реализацию | [L11](../../../../tests/test_chromaseed_refine_precision.py#L11) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_storage_round_trip_preserves_biases_and_bounds_channel_error` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('family', ['stats_mlp', 'patch_mlp', 'recur_soft', 'recur_dynamic'])
def test_storage_round_trip_preserves_biases_and_bounds_channel_error(family):
    from chromaseed_refine import BankNet
    from chromaseed_refine_numpy import NumpyRefiner
    from chromaseed_refine_precision import decode, encode

    payload = {**BankNet(family, [17]).export_slot(0), "x_mean": np.zeros(36, np.float32), "x_std": np.ones(36, np.float32),
               "t_mean": np.zeros(18, np.float32), "t_std": np.ones(18, np.float32), "y_mean": np.zeros(3, np.float32),
               "y_std": np.ones(3, np.float32), "anchor": np.zeros((37, 3), np.float32), "exit_threshold": np.asarray(.5, np.float32)}
    fp16 = encode(payload, "fp16_storage")
    np.testing.assert_array_equal(decode(fp16)["theta"], payload["theta"].astype(np.float16).astype(np.float32))
    q8 = encode(payload, "int8_channels")
    original, restored = NumpyRefiner(payload), NumpyRefiner(decode(q8))
    for name, (weight, bias) in original.layers.items():
        actual_weight, actual_bias = restored.layers[name]
        assert np.all(np.abs(actual_weight - weight) <= q8[f"{name}_scale"][None] / 2 + 1e-7)
        if bias is not None:
            np.testing.assert_array_equal(actual_bias, bias)
    assert np.isfinite(decode(q8)["theta"]).all()
    assert sum(v.nbytes for v in q8.values() if np.issubdtype(v.dtype, np.number)) < payload["theta"].nbytes / 2
```

</details>
