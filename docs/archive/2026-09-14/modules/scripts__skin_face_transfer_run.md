# `scripts/skin_face_transfer_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_transfer_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seg2 real-data preflights and fixed six-trajectory fine-tuning, queued behind HR/P3.

SHA-256 исходника: `195476b991e0145b7ca3b34e93cec3d05686f8c3f8df3e1e6856a45df268e88b`. Строк: **436**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from chromaseed_head_range_verification import process_alive, require_quiet_host
from skin_face_segment import SkinUNet, augment, skin_loss
from skin_face_transfer_data import (
    SOURCES,
    PairedSampler,
    assemble_batch,
    checked,
    digest,
    load_split,
    read,
)
from skin_face_transfer_study import (
    BATCH,
    BUDGETS,
    COUNTS,
    INITIAL,
    INITIAL_SHA,
    INTERVAL,
    PARAMETERS,
    ROOT,
    RUN,
    WIDTH,
    check_bindings,
    freeze_choices,
    freeze_registration,
    utc,
    verify_registration,
    write_once,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 46](../../../../scripts/skin_face_transfer_run.py#L46)

```python
PRIOR = [(Path(f'D:/Luma-RnD/{name}/job.json'), ROOT / f'docs/benchmarks/{name}/verification.json')
         for name in ('chromaseed_head_range_v1', 'chromaseed_palette_transfer_v1')]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `check_predecessor_records` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_face_transfer_run.py#L50) |
| `require_predecessors` | FunctionDef | См. реализацию | [L64](../../../../scripts/skin_face_transfer_run.py#L64) |
| `progress` | FunctionDef | Only noncritical telemetry may be skipped on a transient Windows reader lock. | [L79](../../../../scripts/skin_face_transfer_run.py#L79) |
| `save_npz` | FunctionDef | См. реализацию | [L99](../../../../scripts/skin_face_transfer_run.py#L99) |
| `state_content` | FunctionDef | См. реализацию | [L107](../../../../scripts/skin_face_transfer_run.py#L107) |
| `_load` | FunctionDef | См. реализацию | [L116](../../../../scripts/skin_face_transfer_run.py#L116) |
| `load_initial` | FunctionDef | См. реализацию | [L133](../../../../scripts/skin_face_transfer_run.py#L133) |
| `load_checkpoint` | FunctionDef | См. реализацию | [L137](../../../../scripts/skin_face_transfer_run.py#L137) |
| `save_checkpoint` | FunctionDef | См. реализацию | [L141](../../../../scripts/skin_face_transfer_run.py#L141) |
| `tensor_batch` | FunctionDef | См. реализацию | [L157](../../../../scripts/skin_face_transfer_run.py#L157) |
| `precision` | FunctionDef | См. реализацию | [L167](../../../../scripts/skin_face_transfer_run.py#L167) |
| `update` | FunctionDef | См. реализацию | [L171](../../../../scripts/skin_face_transfer_run.py#L171) |
| `load_training_views` | FunctionDef | См. реализацию | [L210](../../../../scripts/skin_face_transfer_run.py#L210) |
| `setup` | FunctionDef | См. реализацию | [L217](../../../../scripts/skin_face_transfer_run.py#L217) |
| `preflight` | FunctionDef | См. реализацию | [L227](../../../../scripts/skin_face_transfer_run.py#L227) |
| `matching_preflight` | FunctionDef | См. реализацию | [L295](../../../../scripts/skin_face_transfer_run.py#L295) |
| `require_finished` | FunctionDef | См. реализацию | [L311](../../../../scripts/skin_face_transfer_run.py#L311) |
| `run` | FunctionDef | См. реализацию | [L318](../../../../scripts/skin_face_transfer_run.py#L318) |
| `main` | FunctionDef | См. реализацию | [L422](../../../../scripts/skin_face_transfer_run.py#L422) |
