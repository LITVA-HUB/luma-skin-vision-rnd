# `scripts/skin_face_transfer_evaluate.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_transfer_evaluate.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seg2 mask and apparent-image color evaluation, after all validation choices are frozen.

SHA-256 исходника: `7d8d5619eabd8ce9f0bfc497f6dce72025b168f678e233bd7a46bbf1f6ee63d3`. Строк: **251**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import time
import numpy as np
import torch
from chromaseed_head_range_verification import process_alive
from skin_face_appearance import appearance, color_error
from skin_face_transfer_data import SOURCES, checked, digest, load_split, read
from skin_face_transfer_run import (
    load_checkpoint,
    precision,
    require_predecessors,
    save_npz,
    setup,
    tensor_batch,
)
from skin_face_transfer_study import (
    BATCH,
    COUNTS,
    INITIAL,
    INITIAL_SHA,
    RUN,
    check_bindings,
    freeze_choices,
    recipe,
    utc,
    verify_registration,
    write_once,
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_ratio` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_face_transfer_evaluate.py#L36) |
| `mask_summary` | FunctionDef | См. реализацию | [L40](../../../../scripts/skin_face_transfer_evaluate.py#L40) |
| `color_arrays` | FunctionDef | См. реализацию | [L59](../../../../scripts/skin_face_transfer_evaluate.py#L59) |
| `color_summary` | FunctionDef | См. реализацию | [L76](../../../../scripts/skin_face_transfer_evaluate.py#L76) |
| `summary` | FunctionDef | См. реализацию | [L86](../../../../scripts/skin_face_transfer_evaluate.py#L86) |
| `confusion_from_masks` | FunctionDef | См. реализацию | [L93](../../../../scripts/skin_face_transfer_evaluate.py#L93) |
| `unpack_masks` | FunctionDef | См. реализацию | [L101](../../../../scripts/skin_face_transfer_evaluate.py#L101) |
| `verify_mask_arrays` | FunctionDef | См. реализацию | [L111](../../../../scripts/skin_face_transfer_evaluate.py#L111) |
| `evaluate_split` | FunctionDef | См. реализацию | [L120](../../../../scripts/skin_face_transfer_evaluate.py#L120) |
| `validate_selection` | FunctionDef | См. реализацию | [L149](../../../../scripts/skin_face_transfer_evaluate.py#L149) |
| `completed_selection` | FunctionDef | См. реализацию | [L155](../../../../scripts/skin_face_transfer_evaluate.py#L155) |
| `selected_models` | FunctionDef | См. реализацию | [L188](../../../../scripts/skin_face_transfer_evaluate.py#L188) |
| `run_held` | FunctionDef | См. реализацию | [L199](../../../../scripts/skin_face_transfer_evaluate.py#L199) |
| `main` | FunctionDef | См. реализацию | [L243](../../../../scripts/skin_face_transfer_evaluate.py#L243) |
