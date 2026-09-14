# `scripts/skin_correction_transfer_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_correction_transfer_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audited camera-transfer negatives for unchanged stable-color correction.

SHA-256 исходника: `a0aef85749a6ded7bfe49dcb6c6f3255a13827a2298a48957c24425cb813ab1f`. Строк: **81**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_correction_transfer import OUT,RUN,active_lock
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_correction_transfer_report.py#L13) |
