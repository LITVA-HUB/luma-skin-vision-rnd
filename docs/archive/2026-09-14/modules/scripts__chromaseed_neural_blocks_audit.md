# `scripts/chromaseed_neural_blocks_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_blocks_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent NB diagnostic audit; audit validity is not equivalence success.

SHA-256 исходника: `bc9e7de3b0de2d1fcb9c2621261388a48ae53664ab178edbace75ffad7dfc4e6`. Строк: **253**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, direct, full_metrics, verify_normalizers
from chromaseed_neural_blocks_run import (
    NP,
    OUT,
    ROOT,
    RUN,
    SPECS,
    bank_path,
    check_map,
    load_data,
    settings,
)
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_neural_prefix_numpy import Predictor
from skin_local_search_train import roles, sha, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `guards` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_neural_blocks_audit.py#L30) |
| `sparse_parent` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_neural_blocks_audit.py#L46) |
| `compared` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_neural_blocks_audit.py#L61) |
| `main` | FunctionDef | См. реализацию | [L80](../../../../scripts/chromaseed_neural_blocks_audit.py#L80) |
