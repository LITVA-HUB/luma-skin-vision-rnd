# `scripts/cc_v7_external_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_external_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report all frozen methods with paired camera-stratified proxy-cluster intervals.

SHA-256 исходника: `ac4f289a341b15fd4a496af5aa5cb50e63741733e6f6e85467a212a9c614b714`. Строк: **163**.

## Зависимости

```python
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cc_v7_external_lock import BENCH, ROOT, checked_lock, load
from cc_v7_verify import score
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/cc_v7_external_report.py#L18)

```python
PAIRS=[("v7_canonical_teacher_sensor","v7_raw_teacher_sensor"),("v7_gt_sensor","v7_gt_native"),("v7_canonical_teacher_sensor","v2_sog"),("v7_gt_sensor","v2_sog"),("v7_canonical_teacher_sensor","gw_ridge1"),("v7_canonical_teacher_sensor","fourier_ridge_combined")]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `paired_draws` | FunctionDef | См. реализацию | [L21](../../../../scripts/cc_v7_external_report.py#L21) |
| `family_measure` | FunctionDef | См. реализацию | [L36](../../../../scripts/cc_v7_external_report.py#L36) |
| `bootstrap` | FunctionDef | См. реализацию | [L49](../../../../scripts/cc_v7_external_report.py#L49) |
| `mean_or_none` | FunctionDef | См. реализацию | [L87](../../../../scripts/cc_v7_external_report.py#L87) |
| `aggregate` | FunctionDef | См. реализацию | [L91](../../../../scripts/cc_v7_external_report.py#L91) |
| `run` | FunctionDef | См. реализацию | [L99](../../../../scripts/cc_v7_external_report.py#L99) |
