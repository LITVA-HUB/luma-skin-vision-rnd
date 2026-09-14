# `docs/benchmarks/fourier_ridge_v2/scripts/cc_verify_fourier_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/fourier_ridge_v2/scripts/cc_verify_fourier_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent saved-weight replay for V6 and the Fourier representation spike.

SHA-256 исходника: `b9256c17e8f5cd6a68947f9687b8f7b9ccadc73c95cd8b0acac892c630d5a4ad`. Строк: **102**.

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

[Строка 19](../../../../docs/benchmarks/fourier_ridge_v2/scripts/cc_verify_fourier_v2.py#L19)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `independently_error` | FunctionDef | См. реализацию | [L22](../../../../docs/benchmarks/fourier_ridge_v2/scripts/cc_verify_fourier_v2.py#L22) |
| `verify_manifest` | FunctionDef | См. реализацию | [L28](../../../../docs/benchmarks/fourier_ridge_v2/scripts/cc_verify_fourier_v2.py#L28) |
| `run` | FunctionDef | См. реализацию | [L35](../../../../docs/benchmarks/fourier_ridge_v2/scripts/cc_verify_fourier_v2.py#L35) |
