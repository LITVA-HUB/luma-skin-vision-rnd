# `scripts/skin_mskcc_summary_pilot.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_summary_pilot.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen source-only skin Lab controls, no calibration/test endpoint access.

SHA-256 исходника: `aba10001ec746bd56a06fcd29fe8a108dd88462e20e0287399c4408cf1bd90bd`. Строк: **139**.

## Зависимости

```python
import json
import math
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT, PROTOCOL, MANIFEST, sha, load_source
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/skin_mskcc_summary_pilot.py#L20)

```python
OUT = ROOT / 'docs/benchmarks/skin_mskcc_summary_v1'
```

[Строка 21](../../../../scripts/skin_mskcc_summary_pilot.py#L21)

```python
RUN = ROOT / 'experiments/runs/skin_mskcc_summary_v1'
```

[Строка 22](../../../../scripts/skin_mskcc_summary_pilot.py#L22)

```python
COVERAGES = [1, .95, .9, .8, .7, .6]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `summarize` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_mskcc_summary_pilot.py#L25) |
| `candidates` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_mskcc_summary_pilot.py#L36) |
| `main` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_mskcc_summary_pilot.py#L50) |
