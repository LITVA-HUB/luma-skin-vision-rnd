# `scripts/skin_loss_field_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_loss_field_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

All skin-loss fields, negative ablations and target-free grid controls.

SHA-256 исходника: `2bbe89788911aea0a4b042650f19b3a80a41e234e084b26d9b85ac0df3f08894`. Строк: **135**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from luma_skin_vision.color import delta_e00
from skin_loss_field_train import OUT,RUN,PRIOR
from skin_loss_field import ARMS
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import subset,rows,write
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_audit import scalar_de
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `nearest` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_loss_field_report.py#L18) |
| `main` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_loss_field_report.py#L22) |
