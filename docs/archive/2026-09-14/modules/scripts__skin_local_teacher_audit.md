# `scripts/skin_local_teacher_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_teacher_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact replay, reference boundary and fixed-coverage audit of localteacher fits and frozen feature normalization.

SHA-256 исходника: `f4c71467e9ab11b0971049a9e7539847e7c7e186a800ffa7380d952ad4398a6b`. Строк: **101**.

## Зависимости

```python
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_local_teacher_features import load,pack
from skin_mskcc_audit import scalar_de
from skin_capture_model import MODES
from skin_local_teacher_model import LocalTeacherColor
from skin_local_teacher_train import OUT,RUN,PROTOCOL,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `verify_precision_amendment` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_local_teacher_audit.py#L14) |
| `main` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_local_teacher_audit.py#L36) |
