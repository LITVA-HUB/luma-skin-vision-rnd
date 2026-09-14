# `scripts/skin_teacher_readout_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_teacher_readout_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

All chosen source readouts and all candidate risk curves, without test claims.

SHA-256 исходника: `3ec93504485c67e5dacf22d236b8509abb49ea05d8354f4431489c02ccab1b39`. Строк: **93**.

## Зависимости

```python
import csv
import json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT, sha
from skin_teacher_readout import OUT, RUN, ARMS, ALPHAS
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_teacher_readout_report.py#L10) |
