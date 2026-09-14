# `scripts/skin_material_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_material_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report all image material controls with source-selection limits.

SHA-256 исходника: `60338d480bfbfc7a814baec76b66d622b491895c8b1257df0baddcda2941efc3`. Строк: **138**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_material_train import OUT,RUN
from skin_material_model import ARMS
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_material_report.py#L14) |
