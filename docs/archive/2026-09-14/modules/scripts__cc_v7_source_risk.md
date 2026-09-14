# `scripts/cc_v7_source_risk.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_source_risk.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Evaluate frozen standard risk heads on reused source validation, never fit there.

SHA-256 исходника: `7ad6c18a72cd35a34253d8be08fe9b64c8aa663de7273aa991adaf7c2d5370f8`. Строк: **75**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from cc_v2_select import raw_predict, selective_result
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v7_verify import verify_manifest
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.v2 import risk_features_invariant
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/cc_v7_source_risk.py#L19)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `run` | FunctionDef | См. реализацию | [L22](../../../../scripts/cc_v7_source_risk.py#L22) |
