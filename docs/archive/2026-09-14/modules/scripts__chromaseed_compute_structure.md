# `scripts/chromaseed_compute_structure.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_compute_structure.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Count dense linear work in the frozen AS/HR forward graph using meta tensors.

No numerical prediction, training, checkpoint loading, GPU context or latency
measurement is performed. Prefix costs below are arithmetic counterfactuals;
the original forward still executes every registered pass.

SHA-256 исходника: `f9980ec455e3b583fce440e049a9721b4914395426f9809a974603c34aea3b6d`. Строк: **192**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_compute_structure.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 20](../../../../scripts/chromaseed_compute_structure.py#L20)

```python
OUT = Path('D:/Luma-RnD/chromaseed_compute_structure_v1')
```

[Строка 21](../../../../scripts/chromaseed_compute_structure.py#L21)

```python
P3 = Path('D:/Luma-RnD/chromaseed_palette_transfer_v1/registration.json')
```

[Строка 22](../../../../scripts/chromaseed_compute_structure.py#L22)

```python
P3_SHA = '90a9a78bc46f5ca1f40befe0cd04e8861374868d41710c099c7573da7a2add17'
```

[Строка 23](../../../../scripts/chromaseed_compute_structure.py#L23)

```python
ORIGINALS = ('scripts/chromaseed_architecture_scale.py', 'scripts/chromaseed_head_range.py',
             'scripts/chromaseed_refine.py')
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `LinearTrace` | — | [L36](../../../../scripts/chromaseed_compute_structure.py#L36) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_compute_structure.py#L27) |
| `binding` | FunctionDef | См. реализацию | [L32](../../../../scripts/chromaseed_compute_structure.py#L32) |
| `LinearTrace` | ClassDef | Local observer on a scratch meta-model instance; original code is unchanged. | [L36](../../../../scripts/chromaseed_compute_structure.py#L36) |
| `summarize_trace` | FunctionDef | См. реализацию | [L51](../../../../scripts/chromaseed_compute_structure.py#L51) |
| `audit_variant` | FunctionDef | См. реализацию | [L84](../../../../scripts/chromaseed_compute_structure.py#L84) |
| `prepare` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_compute_structure.py#L116) |
| `main` | FunctionDef | См. реализацию | [L165](../../../../scripts/chromaseed_compute_structure.py#L165) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L39–41</summary>

```python
def __init__(self, name, original, inputs, outputs, records):
        self.name, self.original = name, original
        self.inputs, self.outputs, self.records = inputs, outputs, records
```

</details>

<details><summary>__call__ · L43–48</summary>

```python
def __call__(self, value):
        if value.device.type != 'meta' or value.shape[-1] != self.inputs:
            raise ValueError('operation audit accepts only matching meta tensors')
        self.records.append(dict(layer=self.name, input_shape=list(value.shape),
                                 macs=math.prod(value.shape[:-1])*self.inputs*self.outputs))
        return self.original(value)
```

</details>
