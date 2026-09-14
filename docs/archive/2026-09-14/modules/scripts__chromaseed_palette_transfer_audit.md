# `scripts/chromaseed_palette_transfer_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_transfer_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent full P3 initialization, rows, exports, selections and all-pass audit.

SHA-256 исходника: `22abb91db2a60ffd7c2fda76cc7bec304b614ecdc100d203c92e3f8dae6f7c44`. Строк: **531**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from chromaseed_gated_audit import model_from
from chromaseed_head_range import Predictor, predict
from chromaseed_head_range_audit import bank_path as prior_bank
from chromaseed_head_range_audit import check_model as check_hr_model
from chromaseed_kernel_audit import nz
from chromaseed_local_denoise_audit import close, verify_normalizers
from chromaseed_palette_transfer_verification import (
    ARMS,
    CONTRACT,
    ENCODER_PARAMETERS,
    EXPECTED,
    FILES,
    HR_OUT,
    HR_RUN,
    NATIVE_INPUTS,
    OUT,
    PARAMETERS,
    RATES,
    REGISTRATION,
    ROLES,
    ROOT,
    RUN,
    SEEDS,
    TIMES,
    check_hashes,
    compare,
    digest,
    encoder_content,
    encoders_for,
    index_records,
    initial_theta,
    primary_gate,
    read,
    registration_check,
    remember,
    select_policies,
    selected_heads,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from chromaseed_widen_run import load_data
from skin_local_search_train import folds_for, roles
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bank_path` | FunctionDef | См. реализацию | [L62](../../../../scripts/chromaseed_palette_transfer_audit.py#L62) |
| `inspect` | FunctionDef | См. реализацию | [L73](../../../../scripts/chromaseed_palette_transfer_audit.py#L73) |
| `check_model` | FunctionDef | См. реализацию | [L165](../../../../scripts/chromaseed_palette_transfer_audit.py#L165) |
| `check_bank` | FunctionDef | См. реализацию | [L182](../../../../scripts/chromaseed_palette_transfer_audit.py#L182) |
| `verify_passes` | FunctionDef | См. реализацию | [L197](../../../../scripts/chromaseed_palette_transfer_audit.py#L197) |
| `candidates_for` | FunctionDef | См. реализацию | [L207](../../../../scripts/chromaseed_palette_transfer_audit.py#L207) |
| `audit_final` | FunctionDef | См. реализацию | [L263](../../../../scripts/chromaseed_palette_transfer_audit.py#L263) |
| `probe_initializers` | FunctionDef | Actual encoders/native FIT statistics, no native fitting and no production seal. | [L314](../../../../scripts/chromaseed_palette_transfer_audit.py#L314) |
| `main` | FunctionDef | См. реализацию | [L401](../../../../scripts/chromaseed_palette_transfer_audit.py#L401) |
