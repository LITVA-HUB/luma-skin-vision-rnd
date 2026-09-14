# `scripts/skin_risk_cross_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_risk_cross_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate all fixed crossings with matched-compute descriptive comparisons.

SHA-256 исходника: `411510d21a9f21a7025108b1dff9e5f358d2333e44da5633d155f73e3005c5d5`. Строк: **98**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_risk_cross import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_risk_cross_report.py#L13) |
