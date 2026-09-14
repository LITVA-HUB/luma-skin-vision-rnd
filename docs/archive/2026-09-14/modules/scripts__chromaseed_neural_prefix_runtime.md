# `scripts/chromaseed_neural_prefix_runtime.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_prefix_runtime.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

NP consumer speed and full original ND training plus exact prefix export.

SHA-256 исходника: `7c39f55487449a01922670070d2a70549f6865a316270a063ed91665a215831f`. Строк: **229**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_gaussian_audit import actual_consumer
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_fit import fit
from chromaseed_local_denoise_numpy import Predictor as NDPredictor
from chromaseed_neural_prefix_numpy import Predictor, export_prefix, predict
from chromaseed_neural_prefix_run import ND, OUT, ROOT, RUN, check_map, load_data
from skin_local_search_train import roles, sha, weights_for, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `time_consumer` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_neural_prefix_runtime.py#L17) |
| `main` | FunctionDef | См. реализацию | [L51](../../../../scripts/chromaseed_neural_prefix_runtime.py#L51) |
