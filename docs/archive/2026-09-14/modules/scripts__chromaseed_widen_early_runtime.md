# `scripts/chromaseed_widen_early_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_widen_early_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

WE consumer timing and full selected six-slot training reconstruction.

SHA-256 исходника: `adb28ed38579d16dd14e2a82876e0857b23e5f10b3ed8e621e663a1a9675842f`. Строк: **213**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit as upstream_fit
from chromaseed_long_training_run import ND, NP, context
from chromaseed_neural_prefix_numpy import export_prefix
from chromaseed_refine_train import setup
from chromaseed_widen import SEEDS, predict
from chromaseed_widen_audit import exact
from chromaseed_widen_early_fit import fit
from chromaseed_widen_early_run import CAPS, OUT, ROOT, RUN, check_map, load_data
from chromaseed_widen_runtime import time_response
from skin_local_search_train import sha, weights_for, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_widen_early_runtime.py#L22) |
