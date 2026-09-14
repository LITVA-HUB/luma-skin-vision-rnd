# `scripts/skin_issa_material.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_issa_material.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed oracle compression controls for real measured skin reflectance.

SHA-256 исходника: `cef1a5bbef5bc54baefc9ad7872f51a5e18341fa9879d14dae832ab9e9e279f6`. Строк: **172**.

## Зависимости

```python
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from scripts.skin_issa_data import OUT, PRIVATE, sha, write_json, checked_manifest
from scripts.skin_issa_color import spectral_xyz, source_lab
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/skin_issa_material.py#L10)

```python
ROOT=Path(__file__).resolve().parents[1]
```

[Строка 16](../../../../scripts/skin_issa_material.py#L16)

```python
METHODS=('reflectance','density','logit')
```

[Строка 17](../../../../scripts/skin_issa_material.py#L17)

```python
WIDTHS=(2,3,4,6,8)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit_basis` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_issa_material.py#L20) |
| `subject_weights` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_issa_material.py#L31) |
| `encode` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_issa_material.py#L36) |
| `decode` | FunctionDef | См. реализацию | [L45](../../../../scripts/skin_issa_material.py#L45) |
| `load_cache` | FunctionDef | См. реализацию | [L52](../../../../scripts/skin_issa_material.py#L52) |
| `train` | FunctionDef | См. реализацию | [L65](../../../../scripts/skin_issa_material.py#L65) |
| `verify_lock` | FunctionDef | См. реализацию | [L91](../../../../scripts/skin_issa_material.py#L91) |
| `stats` | FunctionDef | См. реализацию | [L96](../../../../scripts/skin_issa_material.py#L96) |
| `evaluate` | FunctionDef | См. реализацию | [L101](../../../../scripts/skin_issa_material.py#L101) |
