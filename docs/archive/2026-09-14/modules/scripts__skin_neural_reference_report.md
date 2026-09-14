# `scripts/skin_neural_reference_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_neural_reference_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate audited adapter fits, full risk curves and in-sample residual diagnostic.

SHA-256 исходника: `06294221c01d36ea3a01af5c904f5d66e8080a30e2c009bc77e2456df987afa4`. Строк: **118**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from skin_neural_reference_train import OUT,RUN
from skin_neural_reference import ColorAdapter
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_neural_reference_report.py#L15) |
