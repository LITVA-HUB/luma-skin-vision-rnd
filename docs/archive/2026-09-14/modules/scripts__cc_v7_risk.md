# `scripts/cc_v7_risk.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_risk.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched V7 error heads fitted only to held-out source RISK/CAL predictions.

SHA-256 исходника: `8fd7f2ccc76b5b0da4e626c4c624fca7d143d8c9238ff40b0b8ca4c0b90f6fdb`. Строк: **127**.

## Зависимости

```python
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from cc_v2_select import estimator, raw_predict
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_verify import verify_manifest
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from luma_skin_vision.cc.core import reproduction, selective_curve
from luma_skin_vision.cc.v2 import CompactResidualCC, risk_features_invariant
from luma_skin_vision.cc.v2_experiment import split_indices
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/cc_v7_risk.py#L23)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read_in_role_order` | FunctionDef | См. реализацию | [L26](../../../../scripts/cc_v7_risk.py#L26) |
| `predict` | FunctionDef | См. реализацию | [L33](../../../../scripts/cc_v7_risk.py#L33) |
| `fit_heads` | FunctionDef | См. реализацию | [L44](../../../../scripts/cc_v7_risk.py#L44) |
| `run` | FunctionDef | См. реализацию | [L80](../../../../scripts/cc_v7_risk.py#L80) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L33–41</summary>

```python
def predict(model,x,batch=32):
    arrays={k:[] for k in ("pred","context","cheap","valid")}
    for start in range(0,len(x),batch):
        xb=x[start:start+batch]
        pred,context=model(xb)
        output={"pred":pred,**risk_features_invariant(xb,pred,context)}
        for key in arrays:
            arrays[key].append(output[key].numpy())
    return {k:np.concatenate(v) for k,v in arrays.items()}
```

</details>
