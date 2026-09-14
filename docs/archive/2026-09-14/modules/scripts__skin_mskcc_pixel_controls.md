# `scripts/skin_mskcc_pixel_controls.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_pixel_controls.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Nine predeclared local-pixel controls; source validation only.

SHA-256 исходника: `9959228a387d33e1a90495f3083afa54e28b57f420a14ac29bc0864990f09de4`. Строк: **58**.

## Зависимости

```python
import json
import warnings
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load,PROTOCOL
from skin_mskcc_summary_pilot import summarize
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_mskcc_pixel_controls.py#L19) |
