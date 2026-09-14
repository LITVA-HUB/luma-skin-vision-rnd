# `scripts/skin_teacher_conflict.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_teacher_conflict.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Post-hoc source gradient diagnostic; no fitting or parameter mutation.

SHA-256 исходника: `ffb72a062e597b03567ca886a9186a7022b5b3ca251b057b71f250f3c839f112`. Строк: **76**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT, sha
from skin_mskcc_train_pixels import digest
from skin_local_teacher_features import load, pack, OUT
from skin_local_teacher_model import LocalTeacherColor
from skin_capture_model import MODES
from skin_pair_invariance import pair_indices
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `gradient_relation` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_teacher_conflict.py#L14) |
| `main` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_teacher_conflict.py#L27) |
