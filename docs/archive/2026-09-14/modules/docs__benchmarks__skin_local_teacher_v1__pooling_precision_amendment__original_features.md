# `docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen spatially corresponding skin teacher tokens; source caches only.

SHA-256 исходника: `e0972a6ba99ac0e251264d3584f203fa748c8336b80505a981bc6317c1a0498c`. Строк: **110**.

## Зависимости

```python
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load as load_pixels
from skin_teacher_features import teacher, preprocess, TEACHER
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L13)

```python
OUT = ROOT/'docs/benchmarks/skin_local_teacher_v1'
```

[Строка 14](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L14)

```python
CACHE = ROOT/'data/processed/skin_local_teacher_v1'
```

[Строка 15](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L15)

```python
PROTOCOL = ROOT/'docs/research/skin_local_teacher_protocol_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `pool_tokens` | FunctionDef | См. реализацию | [L18](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L18) |
| `image_permutation` | FunctionDef | См. реализацию | [L24](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L24) |
| `extract` | FunctionDef | См. реализацию | [L29](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L29) |
| `load` | FunctionDef | См. реализацию | [L44](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L44) |
| `pack` | FunctionDef | См. реализацию | [L58](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L58) |
| `normalization` | FunctionDef | См. реализацию | [L66](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L66) |
| `main` | FunctionDef | См. реализацию | [L72](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_amendment/original_features.py#L72) |
