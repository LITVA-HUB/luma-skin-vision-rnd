# `scripts/chromaseed_palette_transfer_preflight.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_transfer_preflight.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Synthetic P3 training gates using actual P2 encoders; never native quality.

SHA-256 исходника: `1c5702214029e400606c18e4f439f9525f7199c80908bccc860cbe5f17327743`. Строк: **194**.

## Зависимости

```python
from __future__ import annotations
import argparse
import gc
import sys
import time
import numpy as np
import torch
from chromaseed_head_range import Predictor, predict_torch
from chromaseed_head_range_fit import fit as original_fit
from chromaseed_head_range_verification import (
    check_hashes,
    digest,
    read,
    require_quiet_host,
    write_once,
)
from chromaseed_palette_transfer import ARMS, VARIANTS
from chromaseed_palette_transfer_fit import fit
from chromaseed_palette_transfer_run import (
    ROOT,
    RUN,
    encoders_for,
    require_hr_sealed,
    verify_registration,
)
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_palette_transfer_preflight.py#L35) |
