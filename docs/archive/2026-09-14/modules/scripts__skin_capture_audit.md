# `scripts/skin_capture_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_capture_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only replay and fit-boundary audit of the fixed 54-run factorial.

SHA-256 исходника: `e629b1f1acb794b2049c64b6b5ba129100fd66cb3213f3350d73afa324ef3230`. Строк: **53**.

## Зависимости

```python
import json
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_capture_model import CaptureColor,MODES
from skin_capture_train import OUT,RUN,PROTOCOL,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_capture_audit.py#L12) |
