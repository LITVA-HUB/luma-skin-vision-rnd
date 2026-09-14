# `docs/benchmarks/cc_v6/scripts/cc_verify_new_screens.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v6/scripts/cc_verify_new_screens.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent saved-weight replay for V6 and the Fourier representation spike.

SHA-256 исходника: `f1f3c867731522ffdc0ffb306177753ab1585a02ccd1442ac65971339cf2254a`. Строк: **102**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cc_fourier_ridge import predict_score
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v3_ffcc import decode, featurize
from cc_v6_evaluate import evaluate
from cc_v6_model import CanonicalEvidenceNet
from threadpoolctl import threadpool_limits
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../docs/benchmarks/cc_v6/scripts/cc_verify_new_screens.py#L19)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `independently_error` | FunctionDef | См. реализацию | [L22](../../../../docs/benchmarks/cc_v6/scripts/cc_verify_new_screens.py#L22) |
| `verify_manifest` | FunctionDef | См. реализацию | [L28](../../../../docs/benchmarks/cc_v6/scripts/cc_verify_new_screens.py#L28) |
| `run` | FunctionDef | См. реализацию | [L35](../../../../docs/benchmarks/cc_v6/scripts/cc_verify_new_screens.py#L35) |
