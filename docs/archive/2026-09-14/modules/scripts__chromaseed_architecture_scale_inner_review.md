# `scripts/chromaseed_architecture_scale_inner_review.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_inner_review.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Lightweight provisional review of completed inner folds; never opens final outputs.

SHA-256 исходника: `f63444f8cb71133cef6488fa1ed88015aeb45f6e4fd424bb1677dcefab52ed05`. Строк: **107**.

## Зависимости

```python
from __future__ import annotations
import argparse
import numpy as np
from chromaseed_architecture_scale_run import (
    OUT,
    ROOT,
    RUN,
    TIMES,
    VARIANTS,
    WE,
    bank_path,
    load_data,
)
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import roles, sha, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_architecture_scale_inner_review.py#L23) |
