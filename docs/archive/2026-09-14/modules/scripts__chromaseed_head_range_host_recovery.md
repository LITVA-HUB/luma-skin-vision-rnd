# `scripts/chromaseed_head_range_host_recovery.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_host_recovery.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Versioned HR runtime host adaptation; original numerical sources remain frozen.

SHA-256 исходника: `73676516926c66cabcb5d63c93d6754fae428ed61fe9b6b1eefce5b0979b0e38`. Строк: **355**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from chromaseed_head_range_verification import (
    CONTRACT,
    OUT,
    ROOT,
    RUN,
    check_hashes,
    digest,
    process_alive,
    read,
    require_terminal,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_windows_host_guard import assess, collect_snapshot, make_registry
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 28](../../../../scripts/chromaseed_head_range_host_recovery.py#L28)

```python
RECOVERY = RUN / "verification_v1/host_recovery_v1"
```

[Строка 29](../../../../scripts/chromaseed_head_range_host_recovery.py#L29)

```python
PROTOCOL = RECOVERY / "protocol.json"
```

[Строка 30](../../../../scripts/chromaseed_head_range_host_recovery.py#L30)

```python
AUDIT_SHA = "283dad925b9f8ed36b269418fc935854d0fb3b2d396e26e4f0b760e0fb9e90df"
```

[Строка 31](../../../../scripts/chromaseed_head_range_host_recovery.py#L31)

```python
FILES = (
    "scripts/chromaseed_windows_host_guard.py",
    "scripts/chromaseed_head_range_host_recovery.py",
    "tests/test_chromaseed_windows_host_guard.py",
    "tests/test_chromaseed_head_range_host_recovery.py",
    "docs/research/chromaseed_head_range_host_recovery_v1_protocol.md",
)
```

[Строка 38](../../../../scripts/chromaseed_head_range_host_recovery.py#L38)

```python
NUMERICAL = (
    "response",
    "replay",
    "fit",
    "upstream_fit",
    "Predictor",
    "predict_torch",
    "compare",
    "exact",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `utc` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_head_range_host_recovery.py#L50) |
| `install_runtime` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_head_range_host_recovery.py#L54) |
| `render_disclosure` | FunctionDef | См. реализацию | [L85](../../../../scripts/chromaseed_head_range_host_recovery.py#L85) |
| `supplement` | FunctionDef | См. реализацию | [L99](../../../../scripts/chromaseed_head_range_host_recovery.py#L99) |
| `freeze` | FunctionDef | См. реализацию | [L105](../../../../scripts/chromaseed_head_range_host_recovery.py#L105) |
| `guard_evidence` | FunctionDef | См. реализацию | [L194](../../../../scripts/chromaseed_head_range_host_recovery.py#L194) |
| `run` | FunctionDef | См. реализацию | [L223](../../../../scripts/chromaseed_head_range_host_recovery.py#L223) |
| `verify` | FunctionDef | См. реализацию | [L305](../../../../scripts/chromaseed_head_range_host_recovery.py#L305) |
| `report` | FunctionDef | См. реализацию | [L327](../../../../scripts/chromaseed_head_range_host_recovery.py#L327) |
