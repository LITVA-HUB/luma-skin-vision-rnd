# `scripts/chromaseed_neural_blocks_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_blocks_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Complete paired NB fit costs; correctness failures remain explicit.

SHA-256 исходника: `b884881e8e81833bf9ce9eeafd34fa8a464c1cdf8bd4c17e650fd350daeea094`. Строк: **174**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as full_fit
from chromaseed_neural_blocks_diagnose import array_guards
from chromaseed_neural_blocks_fit import fit as subset_fit
from chromaseed_neural_blocks_fit import to_prefix
from chromaseed_neural_blocks_run import NP, OUT, ROOT, RUN, check_map, load_data, outputs, settings
from chromaseed_neural_prefix_numpy import export_prefix
from skin_local_search_train import metrics, roles, sha, weights_for, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `checked` | FunctionDef | См. реализацию | [L18](../../../../scripts/chromaseed_neural_blocks_runtime.py#L18) |
| `main` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_neural_blocks_runtime.py#L39) |
