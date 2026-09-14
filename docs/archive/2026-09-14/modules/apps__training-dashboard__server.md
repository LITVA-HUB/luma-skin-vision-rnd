# `apps/training-dashboard/server.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../apps/training-dashboard/server.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Loopback-only static dashboard and read-only JSON telemetry API.

SHA-256 исходника: `0f327dabc1cfc127d68013c7aa65453a4938b7b5a0dbea5230d5aa02fac8acf6`. Строк: **158**.

## Зависимости

```python
import argparse
import csv
import io
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from face_transfer_monitor import snapshot as face_transfer_snapshot
from facial_monitor import snapshot as facial_snapshot
from head_range_monitor import snapshot as head_range_snapshot
from monitor import Monitor
from palette_transfer_monitor import snapshot as palette_transfer_snapshot
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `export_csv` | FunctionDef | См. реализацию | [L20](../../../../apps/training-dashboard/server.py#L20) |
| `make_handler` | FunctionDef | См. реализацию | [L59](../../../../apps/training-dashboard/server.py#L59) |
| `main` | FunctionDef | См. реализацию | [L138](../../../../apps/training-dashboard/server.py#L138) |
