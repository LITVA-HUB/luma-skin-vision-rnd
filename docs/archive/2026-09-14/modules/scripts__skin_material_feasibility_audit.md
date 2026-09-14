# `scripts/skin_material_feasibility_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_material_feasibility_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audit achieved oracle codes; no claim of globally optimal infeasibility.

SHA-256 исходника: `1b637dd673b6188e7cea02ddd0ee2252176f509d2ab7cc3c43a3b49a1986f59c`. Строк: **36**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from skin_material_prior import PRIOR,material_value_jacobian
from skin_material_train import OUT,RUN
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_material_feasibility_audit.py#L13) |
