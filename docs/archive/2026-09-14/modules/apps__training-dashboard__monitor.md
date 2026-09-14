# `apps/training-dashboard/monitor.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../apps/training-dashboard/monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only observability for the existing AS run; no ML imports or writers.

SHA-256 исходника: `ba889ad1e430fda1286db6109c26356ef60cb5ec3b7e6f9e4d465a7cedf9eeb8`. Строк: **420**.

## Зависимости

```python
from __future__ import annotations
import csv
import ctypes
import json
import math
import os
import statistics
import subprocess
import time
from collections import deque
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../apps/training-dashboard/monitor.py#L16)

```python
VARIANTS = (
    "patch_small",
    "patch5m",
    "soft_small",
    "soft5m",
    "dynamic_small",
    "dynamic5m",
    "pool5m",
)
```

[Строка 25](../../../../apps/training-dashboard/monitor.py#L25)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 26](../../../../apps/training-dashboard/monitor.py#L26)

```python
LABELS = {
    "patch_small": "Участки · малая",
    "patch5m": "Участки · 4,96 млн",
    "soft_small": "Уточнения · малая",
    "soft5m": "Уточнения · 4,85 млн",
    "dynamic_small": "Динамическая · малая",
    "dynamic5m": "Динамическая · 4,85 млн",
    "pool5m": "Объединение признаков · 4,85 млн",
}
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Monitor` | — | [L98](../../../../apps/training-dashboard/monitor.py#L98) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `process_alive` | FunctionDef | Check the actual worker, rather than trusting a stale job.json. | [L37](../../../../apps/training-dashboard/monitor.py#L37) |
| `query_gpu` | FunctionDef | См. реализацию | [L64](../../../../apps/training-dashboard/monitor.py#L64) |
| `Monitor` | ClassDef | См. реализацию | [L98](../../../../apps/training-dashboard/monitor.py#L98) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L99–106</summary>

```python
def __init__(self, root, clock=time.time, process_probe=process_alive, gpu_probe=query_gpu):
        self.root = Path(root)
        self.run = self.root / "experiments/runs/chromaseed_architecture_scale_v1"
        self.out = self.root / "docs/benchmarks/chromaseed_architecture_scale_v1"
        self.clock, self.process_probe, self.gpu_probe = clock, process_probe, gpu_probe
        self.cache, self.warnings = {}, []
        self.gpu_at, self.gpu = 0, {"available": False}
        self.gpu_history = deque(maxlen=600)
```

</details>
