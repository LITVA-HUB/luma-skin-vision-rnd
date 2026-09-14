# `scripts/chromaseed_recurrent_stream.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_recurrent_stream.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Lazy batch-one AS/HR recurrence, with explicitly bounded execution.

Consumes exported features and weights only; no Torch or image processing.
The default executes all four trained passes. An optional positive L2 threshold
compares successive native Lab predictions: it is NOT a calibrated error bound,
confidence score, or an adopted early-exit policy. No payload exit threshold is
inherited. Validate any prospective policy on INNER data before evaluating it.

SHA-256 исходника: `fda76f61d3d7074b634bebec0bb9939a5ee718b34ed97047ab749611c3891730`. Строк: **185**.

## Зависимости

```python
from __future__ import annotations
import math
from dataclasses import dataclass
from numbers import Real
import numpy as np
from scipy.special import expit
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_recurrent_stream.py#L19)

```python
VARIANTS = ("soft_small", "dynamic_small", "soft5m", "dynamic5m")
```

[Строка 20](../../../../scripts/chromaseed_recurrent_stream.py#L20)

```python
MODES = ("unit", "wide", "linear")
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `PassOutput` | — | [L24](../../../../scripts/chromaseed_recurrent_stream.py#L24) |
| `PredictionTrace` | — | [L32](../../../../scripts/chromaseed_recurrent_stream.py#L32) |
| `StreamingPredictor` | — | [L59](../../../../scripts/chromaseed_recurrent_stream.py#L59) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `PassOutput` | ClassDef | См. реализацию | [L24](../../../../scripts/chromaseed_recurrent_stream.py#L24) |
| `PredictionTrace` | ClassDef | См. реализацию | [L32](../../../../scripts/chromaseed_recurrent_stream.py#L32) |
| `_array` | FunctionDef | См. реализацию | [L45](../../../../scripts/chromaseed_recurrent_stream.py#L45) |
| `_silu` | FunctionDef | См. реализацию | [L55](../../../../scripts/chromaseed_recurrent_stream.py#L55) |
| `StreamingPredictor` | ClassDef | One region per call; generators share immutable weights, not request state. | [L59](../../../../scripts/chromaseed_recurrent_stream.py#L59) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L62–95</summary>

```python
def __init__(self, model):
        self.variant = str(model["variant"])
        self.mode = str(model.get("head_mode", "unit"))
        if self.variant not in VARIANTS or self.mode not in MODES:
            raise ValueError("only recurrent AS/HR exports and unit/wide/linear heads are supported")
        a, e, h = (32, 24, 48) if self.variant.endswith("small") else (384, 256, 1120)
        layout = (
            ("token1", 18, a), ("token2", a, e), ("context", 36 + e, h),
            ("query", h + 3, e), ("key", e, e),
            ("update1", 2 * h + e + 3, h), ("update2", h, h), ("head", h, 3),
        )
        count = sum((incoming + (name != "key")) * outgoing
                    for name, incoming, outgoing in layout)
        theta = _array(model["theta"], (count,), "theta").astype(np.float64)
        theta.setflags(write=False)
        self.layers = {}
        offset = 0
        for name, incoming, outgoing in layout:
            size = incoming * outgoing
            weight = theta[offset:offset+size].reshape(incoming, outgoing)
            offset += size
            bias = np.zeros(outgoing) if name == "key" else theta[offset:offset+outgoing]
            offset += 0 if name == "key" else outgoing
            bias.setflags(write=False)
            self.layers[name] = weight, bias
        self.base = {}
        for name, shape in (
            ("x_mean", (36,)), ("x_std", (36,)), ("y_mean", (3,)), ("y_std", (3,)),
            ("w0", (36, 16)), ("b0", (16,)), ("v0", (16, 3)), ("c0", (3,)),
        ):
            self.base[name] = _array(model["base_" + name], shape, name,
                                     positive=name.endswith("_std"))
        self.t_mean = _array(model["t_mean"], (18,), "t_mean")
        self.t_std = _array(model["t_std"], (18,), "t_std", positive=True)
```

</details>

<details><summary>predict · L184–185</summary>

```python
def predict(self, color, patches, **policy):
        return self.predict_trace(color, patches, **policy).prediction
```

</details>
