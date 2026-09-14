# `scripts/skin_appearance_inverse_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_appearance_inverse_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-selected inverse results and separate representation diagnostics.

SHA-256 исходника: `6292c5940878a9965f2bd420b985b61985f26fe61bcc7ca996faa662e1501c81`. Строк: **141**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_appearance_inverse_train import OUT,RUN,ALPHAS
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_appearance_inverse_report.py#L15) |
