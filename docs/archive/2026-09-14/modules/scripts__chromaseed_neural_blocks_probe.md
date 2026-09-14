# `scripts/chromaseed_neural_blocks_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_blocks_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Causal isolation of first-step shape effects, including identical-gradient injection.

SHA-256 исходника: `4a13f8eeb9aaf90edce34bff4265eb07006b0922c7ac80718743805eec5e3f93`. Строк: **118**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
import torch
from chromaseed_kernel_audit import js
from chromaseed_local_denoise import Bank, preprocessor
from chromaseed_local_denoise_fit import noise_sequences
from chromaseed_neural_blocks_fit import RetainedBank
from chromaseed_neural_blocks_run import OUT, ROOT, RUN, SEEDS, check_map, load_data, settings
from chromaseed_refine import BankAdamW
from chromaseed_refine_train import sampling_indices, setup
from skin_local_search_train import roles, sha, weights_for, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `difference` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_neural_blocks_probe.py#L17) |
| `main` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_neural_blocks_probe.py#L22) |
