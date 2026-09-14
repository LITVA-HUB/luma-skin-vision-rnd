# `scripts/skin_color_sampling_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_color_sampling_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched allocation results and person-mass falsifier; no endpoint selection.

SHA-256 исходника: `594a9eaa81d956535d800dc35d3fd0486ff7d25714b45532bdb70550d8e78878`. Строк: **106**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_color_sampling_train import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_color_sampling_report.py#L13) |
