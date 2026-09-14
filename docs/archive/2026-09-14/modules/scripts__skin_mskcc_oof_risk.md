# `scripts/skin_mskcc_oof_risk.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_oof_risk.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

18subject-excluded color fits produce honest TRAIN residual labels for risk.

SHA-256 исходника: `141a195ce77389434b50c1fa749f998011f7d7fbc8d507b06961fa4817609dda`. Строк: **66**.

## Зависимости

```python
import os
import json
import time
import numpy as np
import torch
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import ROOT,OUT,RUN,PROTOCOL,SEEDS,patient_folds,plain_predict,density
from skin_mskcc_vote import PatchVotes
from skin_mskcc_train_pixels import predict
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_data import sha
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_mskcc_oof_risk.py#L17) |
