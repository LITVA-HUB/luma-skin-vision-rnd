# `scripts/cc_v3_source_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v3_source_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audit source-screen receipts and plot measured development-validation results.

SHA-256 исходника: `8c7f996f517f813c6cc4b1cd0f567e0ec8a2020154c3b4068df02b1c4e69b774`. Строк: **197**.

## Зависимости

```python
import json
import shutil
from pathlib import Path
import matplotlib
import numpy as np
from cc_v2_statistics import read_npz_rows
from luma_skin_vision.cc.core import selective_curve, summarize
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
import matplotlib.pyplot as plt
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `independent_errors` | FunctionDef | См. реализацию | [L19](../../../../scripts/cc_v3_source_report.py#L19) |
| `main` | FunctionDef | См. реализацию | [L33](../../../../scripts/cc_v3_source_report.py#L33) |
