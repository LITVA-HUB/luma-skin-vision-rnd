# `scripts/skin_mskcc_pixel_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_pixel_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audit all pixel checkpoints; aggregate source results without opening test.

SHA-256 исходника: `d6e81aff093653d98b4f03537f985f975ec29164c0ebd6d17f03a38e576d6b90`. Строк: **103**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load,PROTOCOL
from skin_mskcc_vote import PatchVotes
from skin_mskcc_audit import scalar_de
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_train_pixels import predict
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_mskcc_pixel_report.py#L18) |
