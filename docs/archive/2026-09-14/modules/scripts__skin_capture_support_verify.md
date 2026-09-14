# `scripts/skin_capture_support_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_capture_support_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact model/plan replay with independent observed-token provenance checks.

SHA-256 исходника: `4ddeeb05100f8737acaf665dd2806ef8e7545c712a4bc4edf7ebd3d7e8fb5f34`. Строк: **109**.

## Зависимости

```python
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_capture_support_train import OUT,RUN,PROTOCOL,prediction,update_digest
from skin_capture_support import ARMS,make_plan,apply_plan
from skin_capture_model import CaptureColor,MODES
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `verify_plans` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_capture_support_verify.py#L17) |
| `main` | FunctionDef | См. реализацию | [L56](../../../../scripts/skin_capture_support_verify.py#L56) |
