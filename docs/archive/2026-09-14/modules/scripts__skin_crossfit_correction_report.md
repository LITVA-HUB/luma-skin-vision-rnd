# `scripts/skin_crossfit_correction_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_crossfit_correction_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audited internal skin correction results; distinguish improvement from hypothesis.

SHA-256 исходника: `e5549f6df01e00919a6df91fc2e6365c6f6ca31e2078f19930f0314de8b128c3`. Строк: **86**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_crossfit_correction_train import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_crossfit_correction_report.py#L13) |
