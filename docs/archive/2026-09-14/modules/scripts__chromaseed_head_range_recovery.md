# `scripts/chromaseed_head_range_recovery.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_recovery.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Resume frozen HR after a Windows status-file sharing failure; I/O-only adapter.

SHA-256 исходника: `5206242d51e468298fa4196dd8c9d9bb1dfc4713cfd754beb12c86449f9877ff`. Строк: **166**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/chromaseed_head_range_recovery.py#L13)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 14](../../../../scripts/chromaseed_head_range_recovery.py#L14)

```python
RUN = Path('D:/Luma-RnD/chromaseed_head_range_v1')
```

[Строка 15](../../../../scripts/chromaseed_head_range_recovery.py#L15)

```python
OUT = RUN / 'recovery_v1'
```

[Строка 16](../../../../scripts/chromaseed_head_range_recovery.py#L16)

```python
SOURCE_LOCK = 'dfb96f902658009f57d37478ee9e2964c951a2bbce8aaba9acda77ff142652b2'
```

[Строка 17](../../../../scripts/chromaseed_head_range_recovery.py#L17)

```python
DELAYS = (.01, .02, .04, .08, .16, .32) + (.5,)*10
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `retry_writer` | FunctionDef | Retry denied replacement; only the exact progress file can be skipped. | [L20](../../../../scripts/chromaseed_head_range_recovery.py#L20) |
| `sha` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_head_range_recovery.py#L50) |
| `read` | FunctionDef | См. реализацию | [L55](../../../../scripts/chromaseed_head_range_recovery.py#L55) |
| `save_once` | FunctionDef | См. реализацию | [L59](../../../../scripts/chromaseed_head_range_recovery.py#L59) |
| `original_sources` | FunctionDef | См. реализацию | [L65](../../../../scripts/chromaseed_head_range_recovery.py#L65) |
| `freeze` | FunctionDef | См. реализацию | [L75](../../../../scripts/chromaseed_head_range_recovery.py#L75) |
| `resume` | FunctionDef | См. реализацию | [L121](../../../../scripts/chromaseed_head_range_recovery.py#L121) |
