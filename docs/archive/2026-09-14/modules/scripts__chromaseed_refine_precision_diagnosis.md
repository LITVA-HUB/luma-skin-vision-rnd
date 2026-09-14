# `scripts/chromaseed_refine_precision_diagnosis.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_refine_precision_diagnosis.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Reproduce the largest FP16 drift without changing any exit threshold.

SHA-256 исходника: `ed512ed3d0c567a684ec8676149c06b534d1f79f4e2d3d0eb894ffdab5a9f0ce`. Строк: **57**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from chromaseed_refine_audit import read_npz
from chromaseed_refine_numpy import NumpyRefiner
from chromaseed_refine_precision import decode
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `diagnose` | FunctionDef | См. реализацию | [L18](../../../../scripts/chromaseed_refine_precision_diagnosis.py#L18) |
