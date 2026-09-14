# `scripts/cc_v7_external_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_external_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Decode exact frozen INTEL remainder and known-camera official source test.

SHA-256 исходника: `8ea8d95c7772b779ee09bb4b280887aea0d1a1dd1ab050ac9a6614dcfb9fc7bb`. Строк: **85**.

## Зависимости

```python
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES
from cc_v7_external_lock import LOCK, ROOT, checked_lock, load
from luma_skin_vision.cc.core import experts, unit
from luma_skin_vision.cc.data import sample
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/cc_v7_external_data.py#L19)

```python
CACHE=ROOT/"data/processed/cc_v7_external128"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `publisher_pair` | FunctionDef | См. реализацию | [L22](../../../../scripts/cc_v7_external_data.py#L22) |
| `prepare` | FunctionDef | См. реализацию | [L47](../../../../scripts/cc_v7_external_data.py#L47) |
