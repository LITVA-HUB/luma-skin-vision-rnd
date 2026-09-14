# `scripts/skin_gradient_transfer_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_gradient_transfer_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate TRAIN mechanism evidence; never a replacement accuracy table.

SHA-256 исходника: `e9a6aed5fe9d47c3ccdc6f78996d247ed84c72578e2222f2b26d6f70b374a74e`. Строк: **110**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_gradient_transfer_run import OUT
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_gradient_transfer_report.py#L13) |
