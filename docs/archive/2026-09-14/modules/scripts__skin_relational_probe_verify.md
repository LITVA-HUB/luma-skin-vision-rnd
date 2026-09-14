# `scripts/skin_relational_probe_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_relational_probe_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent control enumeration, augmented ridge solves and scalar color.

SHA-256 исходника: `282c27fac5991ccde6d2c1fd247d1fb13908ae8ad40f29573a70b5beb2f32cf0`. Строк: **138**.

## Зависимости

```python
import os
import itertools,json,math
from pathlib import Path
import numpy as np
import torch
from skin_relational_probe_run import OUT,RUN,MODELS,bindings,context_features
from skin_relational_probe import loo_ridge
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `independent_correlation` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_relational_probe_verify.py#L16) |
| `independent_scaled` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_relational_probe_verify.py#L26) |
| `main` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_relational_probe_verify.py#L34) |
