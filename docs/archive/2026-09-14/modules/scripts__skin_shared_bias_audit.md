# `scripts/skin_shared_bias_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_shared_bias_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact replay, reference boundary and fixed-coverage audit of shared-bias fits.

SHA-256 исходника: `118528831da11e418112739614c05a67d400afce56fcc89076d83e9bd4638f41`. Строк: **72**.

## Зависимости

```python
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_capture_model import CaptureColor,MODES
from skin_shared_bias_train import OUT,RUN,PROTOCOL,prediction,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_shared_bias_audit.py#L13) |
