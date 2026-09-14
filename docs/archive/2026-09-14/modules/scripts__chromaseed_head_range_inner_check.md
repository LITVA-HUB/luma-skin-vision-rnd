# `scripts/chromaseed_head_range_inner_check.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_inner_check.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent NumPy check of a fixed snapshot of completed HR inner banks.

SHA-256 исходника: `72f2bd11d4a471bbd0cb0a614d25f00aa4a1ef63a020c6402bbb35e8d5f6b0d8`. Строк: **96**.

## Зависимости

```python
import gc
import numpy as np
import torch
from chromaseed_architecture_scale import base_model
from chromaseed_gated import unpack
from chromaseed_head_range import Predictor
from chromaseed_head_range_run import RUN, TIMES, load_data
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context
from skin_local_search_train import sha, write_json
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L16](../../../../scripts/chromaseed_head_range_inner_check.py#L16) |
