# `scripts/chromaseed_kernel_replay_timing.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_kernel_replay_timing.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

One complete process-level replay, distinct from amortized fit-bank timings.

SHA-256 исходника: `706badc4bf2825e424e7f7f3fd06194bc81333932deb96a657217f6eda30be80`. Строк: **75**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_kernel_replay_timing.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_kernel_replay_timing.py#L17) |
| `main` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_kernel_replay_timing.py#L21) |
