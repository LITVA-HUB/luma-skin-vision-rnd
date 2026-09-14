# `scripts/chromaseed_head_range_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent HR row, sampler, export, all-pass and frozen-selection audit.

SHA-256 исходника: `a2fd8ae5627059d26b1bfc3d783dc339b057b0517b6c1b2b5f2f84a098098435`. Строк: **487**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import time
import numpy as np
import torch
from chromaseed_architecture_scale import base_model
from chromaseed_gated_audit import model_from
from chromaseed_head_range import Predictor, predict
from chromaseed_head_range_verification import (
    AS_OUT,
    AS_RUN,
    CONTRACT,
    FILES,
    MODES,
    OUT,
    PARAMETERS,
    RATES,
    ROLES,
    ROOT,
    RUN,
    SEEDS,
    SOURCE,
    TIMES,
    check_hashes,
    compare,
    digest,
    index_records,
    read,
    remember,
    require_terminal,
    select_policies,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_kernel_audit import nz
from chromaseed_local_denoise_audit import close, verify_normalizers
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
| `bank_path` | FunctionDef | См. реализацию | [L52](../../../../scripts/chromaseed_head_range_audit.py#L52) |
| `inspect` | FunctionDef | См. реализацию | [L65](../../../../scripts/chromaseed_head_range_audit.py#L65) |
| `check_model` | FunctionDef | См. реализацию | [L148](../../../../scripts/chromaseed_head_range_audit.py#L148) |
| `check_bank` | FunctionDef | См. реализацию | [L167](../../../../scripts/chromaseed_head_range_audit.py#L167) |
| `audit_inner` | FunctionDef | См. реализацию | [L173](../../../../scripts/chromaseed_head_range_audit.py#L173) |
| `candidates_for` | FunctionDef | См. реализацию | [L191](../../../../scripts/chromaseed_head_range_audit.py#L191) |
| `audit_final` | FunctionDef | См. реализацию | [L234](../../../../scripts/chromaseed_head_range_audit.py#L234) |
| `probe` | FunctionDef | Fixed completed-inner snapshot; cannot create a main audit or quality result. | [L280](../../../../scripts/chromaseed_head_range_audit.py#L280) |
| `main` | FunctionDef | См. реализацию | [L348](../../../../scripts/chromaseed_head_range_audit.py#L348) |
