# `scripts/skin_face_transfer_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_transfer_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Complete Seg2 artifact checks, paired comparisons and quiet-host CPU response timings.

SHA-256 исходника: `960c865facccc71218ed05e69def9611a071fff6c7ea8509e931330c61ee801f`. Строк: **331**.

## Зависимости

```python
from __future__ import annotations
import argparse
import math
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from skin_face_transfer_data import SOURCES, PairedSampler, checked, digest, load_split, read
from skin_face_transfer_evaluate import (
    color_arrays,
    completed_selection,
    selected_models,
    summary,
    unpack_masks,
    verify_mask_arrays,
)
from skin_face_transfer_run import (
    load_checkpoint,
    load_training_views,
    require_finished,
    require_predecessors,
    tensor_batch,
)
from skin_face_transfer_study import (
    ARMS,
    BATCH,
    BUDGETS,
    INITIAL_SHA,
    PARAMETERS,
    RUN,
    SEEDS,
    check_bindings,
    recipe,
    utc,
    write_once,
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `arrays_from` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_face_transfer_report.py#L44) |
| `paired_color` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_face_transfer_report.py#L50) |
| `validate_trace` | FunctionDef | См. реализацию | [L61](../../../../scripts/skin_face_transfer_report.py#L61) |
| `verify_array_record` | FunctionDef | См. реализацию | [L77](../../../../scripts/skin_face_transfer_report.py#L77) |
| `verify_training_artifacts` | FunctionDef | См. реализацию | [L97](../../../../scripts/skin_face_transfer_report.py#L97) |
| `audit` | FunctionDef | См. реализацию | [L141](../../../../scripts/skin_face_transfer_report.py#L141) |
| `runtime` | FunctionDef | См. реализацию | [L173](../../../../scripts/skin_face_transfer_report.py#L173) |
| `comparisons` | FunctionDef | См. реализацию | [L213](../../../../scripts/skin_face_transfer_report.py#L213) |
| `report` | FunctionDef | См. реализацию | [L256](../../../../scripts/skin_face_transfer_report.py#L256) |
| `main` | FunctionDef | См. реализацию | [L323](../../../../scripts/skin_face_transfer_report.py#L323) |
