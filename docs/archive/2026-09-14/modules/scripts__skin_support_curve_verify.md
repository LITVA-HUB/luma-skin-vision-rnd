# `scripts/skin_support_curve_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_support_curve_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar metrics, source patch provenance and fixed-state replay.

SHA-256 исходника: `b4aa966248f37b50179550d08175da4227e9480443c2f034814d28cac66964c4`. Строк: **108**.

## Зависимости

```python
import os
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from skin_support_curve_train import OUT,RUN,PROTOCOL,bindings,predict
from skin_support_curve import SkinRepresentation,patient_roles,pixel_patches
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
from skin_capture_model import MODES
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `check_summary` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_support_curve_verify.py#L18) |
| `refit` | FunctionDef | См. реализацию | [L28](../../../../scripts/skin_support_curve_verify.py#L28) |
| `main` | FunctionDef | См. реализацию | [L46](../../../../scripts/skin_support_curve_verify.py#L46) |
