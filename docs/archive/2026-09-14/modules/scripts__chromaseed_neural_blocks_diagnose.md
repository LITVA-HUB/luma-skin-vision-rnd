# `scripts/chromaseed_neural_blocks_diagnose.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_blocks_diagnose.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Post-failure NB coverage: measure every guard, never relax or hide failures.

SHA-256 исходника: `f743d7aa80561ec1419f403a34a9a0ba3a425e9b68d188b982bb06a02e11ece1`. Строк: **229**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_gate_stability import summaries
from chromaseed_gated import unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_train import save_npz
from chromaseed_neural_blocks_fit import to_prefix
from chromaseed_neural_blocks_run import (
    CHECKPOINTS,
    NP,
    OUT,
    ROOT,
    RUN,
    SEEDS,
    SPECS,
    bank_path,
    check_map,
    load_data,
    outputs,
    settings,
)
from chromaseed_neural_prefix_numpy import capacity, export_prefix
from skin_local_search_train import metrics, roles, sha, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `array_guards` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_neural_blocks_diagnose.py#L31) |
| `freeze` | FunctionDef | См. реализацию | [L47](../../../../scripts/chromaseed_neural_blocks_diagnose.py#L47) |
| `comparison` | FunctionDef | См. реализацию | [L91](../../../../scripts/chromaseed_neural_blocks_diagnose.py#L91) |
| `main` | FunctionDef | См. реализацию | [L113](../../../../scripts/chromaseed_neural_blocks_diagnose.py#L113) |
