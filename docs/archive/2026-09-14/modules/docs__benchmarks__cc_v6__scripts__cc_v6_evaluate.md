# `docs/benchmarks/cc_v6/scripts/cc_v6_evaluate.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v6/scripts/cc_v6_evaluate.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

V6 evaluation in original camera RGB; actions trained in a residual frame.

SHA-256 исходника: `3c876227ebc5696da51a09c536fe85483fa8ebc58ac1b86fc129be5e3731db4b`. Строк: **46**.

## Зависимости

```python
import numpy as np
import torch
from cc_v4_experiment import action_rgb, oracle_errors, refinement_summary, risk_summary
from luma_skin_vision.cc.core import angular, reproduction, summarize
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `evaluate` | FunctionDef | См. реализацию | [L10](../../../../docs/benchmarks/cc_v6/scripts/cc_v6_evaluate.py#L10) |
