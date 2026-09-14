# `scripts/chromaseed_neural_readout_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_readout_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze NR, fit matched representation/readout banks, select, then evaluate.

SHA-256 исходника: `e6adbd7667671e67a4b389b4505adc716d63d634070ccf7c026e20896f6b55f9`. Строк: **471**.

## Зависимости

```python
from __future__ import annotations
import argparse
import os
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from chromaseed_fast_kernel_train import row_hash, valid_bank
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_gaussian import model_id
from chromaseed_gaussian_train import infer, score
from chromaseed_kernel_audit import js, nz
from chromaseed_kernel_train import atomic_npz
from chromaseed_neural_readout import (
    ALPHAS,
    BASES,
    EPOCHS,
    FAMILIES,
    GROUPS,
    PARAMETERS,
    SEEDS,
    basis_spec,
    choose,
    choose_policy,
    fit_head,
    name_for,
    representation_bank,
)
from skin_local_search_train import (
    CACHE_HASH,
    folds_for,
    metrics,
    roles,
    sha,
    weights_for,
    write_json,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 45](../../../../scripts/chromaseed_neural_readout_train.py#L45)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 46](../../../../scripts/chromaseed_neural_readout_train.py#L46)

```python
TG = ROOT / "experiments/runs/chromaseed_gaussian_v1"
```

[Строка 47](../../../../scripts/chromaseed_neural_readout_train.py#L47)

```python
TG_RECEIPT = "19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `lock_sources` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_neural_readout_train.py#L50) |
| `run_bank` | FunctionDef | См. реализацию | [L123](../../../../scripts/chromaseed_neural_readout_train.py#L123) |
| `fit_stage` | FunctionDef | См. реализацию | [L289](../../../../scripts/chromaseed_neural_readout_train.py#L289) |
| `select_stage` | FunctionDef | См. реализацию | [L314](../../../../scripts/chromaseed_neural_readout_train.py#L314) |
| `evaluate_stage` | FunctionDef | См. реализацию | [L375](../../../../scripts/chromaseed_neural_readout_train.py#L375) |
| `main` | FunctionDef | См. реализацию | [L440](../../../../scripts/chromaseed_neural_readout_train.py#L440) |
