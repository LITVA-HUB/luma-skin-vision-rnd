# `scripts/skin_local_teacher_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_teacher_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate all predeclared local-teacher fits, preserving ordinary comparators.

SHA-256 исходника: `bcd0530198501f3774c2420a15e3478d604fd13b4d6252b7736a640d48fb6ebc`. Строк: **117**.

## Зависимости

```python
import csv
import json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT, sha
from skin_local_teacher_train import OUT, RUN
from skin_local_teacher_model import ARMS
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L11](../../../../scripts/skin_local_teacher_report.py#L11) |
