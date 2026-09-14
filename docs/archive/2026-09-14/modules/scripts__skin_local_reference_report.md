# `scripts/skin_local_reference_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_reference_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Positive person-excluded color signal and failed strong camera comparison.

SHA-256 исходника: `32d4e37a6af2a70a1f08755e646db0022fcd5b1a23f963c45f5370e2bb90070a`. Строк: **136**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_local_reference_run import OUT as SCREEN,RUN as SCREEN_RUN
from skin_local_reference_transfer import OUT as TRANSFER,RUN as TRANSFER_RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_local_reference_report.py#L14) |
