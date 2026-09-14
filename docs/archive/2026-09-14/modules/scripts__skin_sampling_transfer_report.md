# `scripts/skin_sampling_transfer_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_sampling_transfer_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Full known/unseen skin-color report without selecting camera-specific winners.

SHA-256 исходника: `f760a1b47d12d9b49016337841e8c460c2856ceedde2eed48047faaaf477c546`. Строк: **106**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_sampling_transfer_train import OUT,RUN
from skin_sampling_transfer import ARMS
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_sampling_transfer_report.py#L14) |
