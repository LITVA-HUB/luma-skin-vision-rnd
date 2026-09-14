# `scripts/chromaseed_windows_host_guard.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_windows_host_guard.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Recorded WDDM desktop activity guard; no model, optimizer or data operations.

SHA-256 исходника: `de9c70cc93996c6b84bfc0e5b32ef583dde7b8aa15bdd9a0e75bb8653bbff646`. Строк: **239**.

## Зависимости

```python
from __future__ import annotations
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from chromaseed_head_range_verification import competing_workers
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/chromaseed_windows_host_guard.py#L15)

```python
DESKTOP_NAMES = frozenset(
    name.lower()
    for name in (
        "System",
        "dwm.exe",
        "Taskmgr.exe",
        "explorer.exe",
        "CrossDeviceResume.exe",
        "SearchHost.exe",
        "StartMenuExperienceHost.exe",
        "NVIDIA Overlay.exe",
        "msedgewebview2.exe",
        "TextInputHost.exe",
        "ShellExperienceHost.exe",
        "EdgeGameAssist.exe",
        "Happ.exe",
        "SnippingTool.exe",
        "ShellHost.exe",
        "chrome.exe",
        "PhoneExperienceHost.exe",
        "ChatGPT.exe",
        "codex-computer-use-swift.exe",
        "Telegram.exe",
        "msedge.exe",
    )
)
```

[Строка 41](../../../../scripts/chromaseed_windows_host_guard.py#L41)

```python
MAX_ENGINE_PERCENT = 10.0
```

[Строка 42](../../../../scripts/chromaseed_windows_host_guard.py#L42)

```python
MAX_COMPUTE_PERCENT = 0.5
```

[Строка 43](../../../../scripts/chromaseed_windows_host_guard.py#L43)

```python
COUNTER = re.compile(
    r"^pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+_eng_\d+)_engtype_(.+)$", re.I
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `parsed_counter` | FunctionDef | См. реализацию | [L48](../../../../scripts/chromaseed_windows_host_guard.py#L48) |
| `gpu_pids` | FunctionDef | См. реализацию | [L62](../../../../scripts/chromaseed_windows_host_guard.py#L62) |
| `identity` | FunctionDef | См. реализацию | [L75](../../../../scripts/chromaseed_windows_host_guard.py#L75) |
| `make_registry` | FunctionDef | См. реализацию | [L82](../../../../scripts/chromaseed_windows_host_guard.py#L82) |
| `assess` | FunctionDef | См. реализацию | [L101](../../../../scripts/chromaseed_windows_host_guard.py#L101) |
| `collect_snapshot` | FunctionDef | См. реализацию | [L179](../../../../scripts/chromaseed_windows_host_guard.py#L179) |
