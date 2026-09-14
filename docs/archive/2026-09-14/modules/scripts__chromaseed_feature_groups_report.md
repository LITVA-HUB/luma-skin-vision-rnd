# `scripts/chromaseed_feature_groups_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_feature_groups_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report all frozen FG choices and costs without selecting on outer outcomes.

SHA-256 исходника: `e9a5be05104ac86e7f320c49978651c5d67e45e19e6ecccb97dabc40764c6a77`. Строк: **444**.

## Зависимости

```python
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_feature_groups_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_feature_groups_report.py#L17)

```python
RUN = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
```

[Строка 18](../../../../scripts/chromaseed_feature_groups_report.py#L18)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_feature_groups_v1"
```

[Строка 19](../../../../scripts/chromaseed_feature_groups_report.py#L19)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-feature-groups-2026-09-13.md"
```

[Строка 20](../../../../scripts/chromaseed_feature_groups_report.py#L20)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 21](../../../../scripts/chromaseed_feature_groups_report.py#L21)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```

[Строка 22](../../../../scripts/chromaseed_feature_groups_report.py#L22)

```python
GROUPS = (
    "raw36",
    "mean3",
    "median3",
    "central9",
    "mean_std6",
    "quant27",
    "no_corr33",
    "projected16",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_write` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_feature_groups_report.py#L34) |
| `main` | FunctionDef | См. реализацию | [L41](../../../../scripts/chromaseed_feature_groups_report.py#L41) |
