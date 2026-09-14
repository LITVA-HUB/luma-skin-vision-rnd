# `scripts/skin_face_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Prospective LaPa segmentation run with validation-only selection and live progress.

SHA-256 исходника: `5fe50a7ac515d368b4d95bd83268c9e3943521e36a7b317a634b46ea588a60dd`. Строк: **222**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
from skin_face_segment import SkinUNet, augment, scores, skin_loss
from skin_lapa_prepare import file_sha
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/skin_face_train.py#L20)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 21](../../../../scripts/skin_face_train.py#L21)

```python
OUT = DATA_ROOT/'facial_skin_v1'
```

[Строка 22](../../../../scripts/skin_face_train.py#L22)

```python
DATA = DATA_ROOT/'lapa'/'prepared_192'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `setup` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_face_train.py#L25) |
| `batch` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_face_train.py#L36) |
| `evaluate` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_face_train.py#L44) |
| `require_prior_completion` | FunctionDef | См. реализацию | [L67](../../../../scripts/skin_face_train.py#L67) |
| `preflight` | FunctionDef | См. реализацию | [L74](../../../../scripts/skin_face_train.py#L74) |
| `train` | FunctionDef | См. реализацию | [L108](../../../../scripts/skin_face_train.py#L108) |
| `main` | FunctionDef | См. реализацию | [L209](../../../../scripts/skin_face_train.py#L209) |
