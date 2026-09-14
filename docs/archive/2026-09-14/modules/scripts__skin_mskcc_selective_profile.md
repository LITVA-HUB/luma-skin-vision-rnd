# `scripts/skin_mskcc_selective_profile.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_selective_profile.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only profile of the frozen three-model color system and CPU risk heads.

This does not change evaluation code or export a deployment implementation.

SHA-256 исходника: `abc116a66e84af722a4b71f25e3e7d12351a9490a0f3a576c079675b069e028b`. Строк: **62**.

## Зависимости

```python
import json,time
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import OUT,SEEDS,verify_lock,deployment_designs
from skin_mskcc_vote import PatchVotes
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_mskcc_selective_profile.py#L16) |
