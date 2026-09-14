# `scripts/skin_offset_diagnostic_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_offset_diagnostic_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Privileged calibration results, explicitly separate from model accuracy.

SHA-256 исходника: `96c206d8c4cb6dc355a9c9fbe5591c10bb50949eb26c66a1cbf2bba9844b3b0d`. Строк: **90**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_offset_diagnostic import OUT,RUN,SOURCE_RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_offset_diagnostic_report.py#L13) |
