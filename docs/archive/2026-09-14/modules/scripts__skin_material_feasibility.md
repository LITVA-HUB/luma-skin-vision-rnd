# `scripts/skin_material_feasibility.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_material_feasibility.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Target-informed single-material feasibility; no image prediction claim.

SHA-256 исходника: `976a23771191587d7567c3b81f3cb0e380441ea24069860cdb0e599cd9a1e6b1`. Строк: **54**.

## Зависимости

```python
import os
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from skin_material_prior import PRIOR,material_value_jacobian
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_material_train import OUT
from skin_pair_train import write
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_material_feasibility.py#L17) |
