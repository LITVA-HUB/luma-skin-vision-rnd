# `scripts/skin_capture_support_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_capture_support_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

All matched support results and separate teacher-assisted routing diagnostic.

SHA-256 исходника: `f02757ad557bbc9299e467428f2f6bb615706ffe6642271fe4c6468df2166315`. Строк: **127**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_capture_support_train import OUT,RUN
from skin_capture_support import ARMS
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_capture_support_report.py#L14) |
