# `scripts/chromaseed_weight_average.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weight_average.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Equal checkpoint-weight means with explicit same-trajectory provenance.

SHA-256 исходника: `5c89f1465d29ac4b2cfe35907850740cb1ad6fd4b405657c602b01af77be7f04`. Строк: **102**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from chromaseed_neural_prefix_numpy import Predictor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../scripts/chromaseed_weight_average.py#L8)

```python
STEPS = (0, 512, 2048, 8192, 32768, 131072)
```

[Строка 9](../../../../scripts/chromaseed_weight_average.py#L9)

```python
METHODS = ("last", "pair", "prefix", "tail3")
```

[Строка 10](../../../../scripts/chromaseed_weight_average.py#L10)

```python
WEIGHTS = ("w0", "b0", "v0", "c0")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `average` | FunctionDef | См. реализацию | [L13](../../../../scripts/chromaseed_weight_average.py#L13) |
| `recipes` | FunctionDef | См. реализацию | [L45](../../../../scripts/chromaseed_weight_average.py#L45) |
| `choose` | FunctionDef | См. реализацию | [L70](../../../../scripts/chromaseed_weight_average.py#L70) |
| `endpoint_name` | FunctionDef | См. реализацию | [L85](../../../../scripts/chromaseed_weight_average.py#L85) |
| `final_union` | FunctionDef | См. реализацию | [L93](../../../../scripts/chromaseed_weight_average.py#L93) |
