# `scripts/chromaseed_architecture_scale_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

AS actual CPU response and complete selected-bank reconstruction measurements.

SHA-256 исходника: `61c49c93ef5f08ebfb9dc411bcfdc32f1df5bd356ec5560b6d8ba951ae83051c`. Строк: **190**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_architecture_scale import RATES, SEEDS, VARIANTS, Predictor, fit, predict_torch
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, bank_path, check_map, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as upstream_fit
from chromaseed_long_training_run import ND, NP, context
from chromaseed_neural_prefix_numpy import export_prefix
from chromaseed_refine_train import setup
from chromaseed_widen_audit import exact
from skin_local_search_train import sha, weights_for, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `response` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_architecture_scale_runtime.py#L20) |
| `main` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_architecture_scale_runtime.py#L49) |
