# `scripts/skin_mskcc_profile_pixels.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_profile_pixels.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Profile seed17 single models separately from verified original-JPEG input work.

SHA-256 исходника: `ff80f094059cc6d1af8c4930ff35544b354070f37c3c5763d4319d382e600308`. Строк: **48**.

## Зависимости

```python
import json
import time
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from skin_mskcc_data import ROOT,RAW,manifest,sha
from skin_mskcc_pixels import load,decode
from skin_mskcc_vote import PatchVotes
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_mskcc_profile_pixels.py#L12) |
