# `scripts/cc_v7_stress.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_stress.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed virtual-sensor diagnostics; never substitute for real camera evaluation.

SHA-256 исходника: `c296fad60e7e85959d540842e4d65effb2214570dc17f5b67f30a94591a98a81`. Строк: **80**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v7_verify import verify_manifest
from luma_skin_vision.cc.core import angular, reproduction, summarize, unit
from luma_skin_vision.cc.v2 import CompactResidualCC
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/cc_v7_stress.py#L17)

```python
ROOT=Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/cc_v7_stress.py#L18)

```python
MATRICES={"identity":np.eye(3),"gain_red":np.diag([2,1,.5]),"gain_blue":np.diag([.5,1,2]),"mix_small":np.array([[.9,.05,.05],[.05,.9,.05],[.05,.05,.9]]),"mix_train_range":np.array([[.65,.25,.10],[.10,.70,.20],[.20,.10,.70]]),"mix_extrapolation":np.array([[.4,.4,.2],[.2,.5,.3],[.3,.2,.5]])}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `prediction` | FunctionDef | См. реализацию | [L22](../../../../scripts/cc_v7_stress.py#L22) |
| `run` | FunctionDef | См. реализацию | [L26](../../../../scripts/cc_v7_stress.py#L26) |
