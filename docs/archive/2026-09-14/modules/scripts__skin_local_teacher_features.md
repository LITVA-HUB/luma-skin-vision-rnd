# `scripts/skin_local_teacher_features.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_teacher_features.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen spatially corresponding skin teacher tokens; source caches only.

SHA-256 исходника: `a6608cfe2353b4e0667907650dada8da0c3ca74faa1dfd4864a1962d2dca7511`. Строк: **115**.

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

[Строка 13](../../../../scripts/skin_local_teacher_features.py#L13)

```python
OUT = ROOT/'docs/benchmarks/skin_local_teacher_v1'
```

[Строка 14](../../../../scripts/skin_local_teacher_features.py#L14)

```python
CACHE = ROOT/'data/processed/skin_local_teacher_v1'
```

[Строка 15](../../../../scripts/skin_local_teacher_features.py#L15)

```python
PROTOCOL = ROOT/'docs/research/skin_local_teacher_protocol_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `pool_tokens` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_local_teacher_features.py#L18) |
| `image_permutation` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_local_teacher_features.py#L24) |
| `extract` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_local_teacher_features.py#L29) |
| `load` | FunctionDef | См. реализацию | [L49](../../../../scripts/skin_local_teacher_features.py#L49) |
| `pack` | FunctionDef | См. реализацию | [L63](../../../../scripts/skin_local_teacher_features.py#L63) |
| `normalization` | FunctionDef | См. реализацию | [L71](../../../../scripts/skin_local_teacher_features.py#L71) |
| `main` | FunctionDef | См. реализацию | [L77](../../../../scripts/skin_local_teacher_features.py#L77) |
