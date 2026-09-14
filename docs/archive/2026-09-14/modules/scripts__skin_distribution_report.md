# `scripts/skin_distribution_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_distribution_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report all frozen density endpoints without selecting a favorable protocol.

SHA-256 исходника: `bcb6385ed309ef8f59bfbf35d2d79d15a5b39bcea82f205c4bf487b4eaf6ce0e`. Строк: **148**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_distribution_train import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_distribution_report.py#L13) |
