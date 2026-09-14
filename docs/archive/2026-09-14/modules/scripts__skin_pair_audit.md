# `scripts/skin_pair_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_pair_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only source replay, fit-boundary and independent DeltaE00 audit.

SHA-256 исходника: `0c011f96b44333683db9c2251be818d5884e1d6d6caa12c6de6c65a14202db27`. Строк: **63**.

## Зависимости

```python
import json
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_invariance import PairedColor,nuisance_transform
from skin_pair_train import OUT,RUN,PROTOCOL,subset,prediction
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_pair_audit.py#L13) |
