# `docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Conventional color-statistics regression: source-only screen, no novelty claim.

Only official train/validation rows are numerically decoded during the screen.
Other NPY rows are skipped as uninterpreted bytes inside the compressed archive.
Prediction is a separate frozen action and never reads illuminant labels.

SHA-256 исходника: `a989a37d3cab88531f58827bf861fe400b4d13a0ef97b00c61c7b9b07c3c3ac2`. Строк: **511**.

## Зависимости

```python
import argparse
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.core import reproduction, summarize
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 29](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L29)

```python
SCHEMA = "cc-statistics-v1"
```

[Строка 30](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L30)

```python
EPS = 1e-12
```

[Строка 31](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L31)

```python
POWERS = (1, 2, 4, 6, 10)
```

[Строка 32](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L32)

```python
QUANTILES = (10, 25, 50, 75, 90, 95, 99)
```

[Строка 33](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L33)

```python
NAMES = ("ridge1", "ridge10", "ridge100", "hgb7", "hgb15", "extratrees128")
```

[Строка 34](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L34)

```python
STAT_NAMES = [f"minkowski_p{p}" for p in POWERS] + [f"quantile{q}" for q in QUANTILES]
```

[Строка 35](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L35)

```python
FEATURE_COLUMNS = (
    [f"log_{stat}_{channel}" for stat in STAT_NAMES for channel in ("r", "g", "b")]
    + [
        f"patch2x2_{row}{col}_{channel}_over_global_gw"
        for channel in ("r", "g", "b")
        for row in range(2)
        for col in range(2)
    ]
    + [f"spatial_std_{channel}_over_global_gw" for channel in ("r", "g", "b")]
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read_npz_rows` | FunctionDef | Read selected C-order NPY rows without deserializing any excluded row. | [L47](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L47) |
| `featurize` | FunctionDef | 51 fixed per-image features; no learned normalization or camera metadata.  Global statistics include common masked zeros and use natural logs. Direct statistics use a common scalar (mean of the three GW anchors); residual-mode statistics use unnormalized per-channel GW anchors. Both append the same channel-normalized spatial ratios. Quantiles use NumPy linear interpolation. | [L92](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L92) |
| `target_ratios` | FunctionDef | См. реализацию | [L137](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L137) |
| `decode_prediction` | FunctionDef | См. реализацию | [L159](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L159) |
| `model_spec` | FunctionDef | См. реализацию | [L178](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L178) |
| `estimator` | FunctionDef | См. реализацию | [L214](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L214) |
| `_feature_batches` | FunctionDef | См. реализацию | [L236](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L236) |
| `screen` | FunctionDef | См. реализацию | [L241](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L241) |
| `predict_arrays` | FunctionDef | Frozen conventional estimator output plus context for later source-only risk fitting. | [L395](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L395) |
| `predict_frozen` | FunctionDef | См. реализацию | [L409](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L409) |
| `main` | FunctionDef | См. реализацию | [L466](../../../../docs/benchmarks/cc_v5/three_seed_screen/seed43/source_snapshot/scripts/cc_v2_statistics.py#L466) |
