# `scripts/skin_support_curve_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_support_curve_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate the fixed-budget screen without selecting a model on holdout.

SHA-256 исходника: `8f8ec602f35a54cdd83a6817b6a370744c6164b998d90f63a1c969c5472046bc`. Строк: **120**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_support_curve_train import OUT,RUN
from skin_pair_train import write
from skin_mskcc_data import ROOT,sha
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_support_curve_report.py#L13) |
