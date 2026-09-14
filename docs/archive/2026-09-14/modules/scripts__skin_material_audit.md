# `scripts/skin_material_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_material_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Replay real-image fits and independently verify skin color and coverage.

SHA-256 исходника: `6722c7dba6785b3bc53a273f99b03e257a6c18416be325c7a99b905b7b5e32e9`. Строк: **84**.

## Зависимости

```python
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_material_train import OUT,RUN,PROTOCOL,prediction
from skin_material_model import MaterialImage,ARMS
from skin_material_prior import PRIOR
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_material_audit.py#L16) |
