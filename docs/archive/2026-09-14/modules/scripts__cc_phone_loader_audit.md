# `scripts/cc_phone_loader_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_loader_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Development-only audit of frozen Beyond RGB TRAIN scenes; no test scoring.

Read already demosaiced camera RGB as released. Never apply a second Bayer
conversion, black subtraction, white-level division, white balance or CCM.
Patch statistics diagnose reference quality; they are not model accuracy.

SHA-256 исходника: `2d3b3587a69d51dd8e882264a4cd7b49e9e5e0342337c001a0f4bf0be8720abc`. Строк: **168**.

## Зависимости

```python
import argparse
import json
from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/cc_phone_loader_audit.py#L18)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 19](../../../../scripts/cc_phone_loader_audit.py#L19)

```python
PROVENANCE = ROOT/'docs/data/provenance/mobile_screen_2026_09_11'
```

[Строка 20](../../../../scripts/cc_phone_loader_audit.py#L20)

```python
LOCK = PROVENANCE/'beyond_rgb_selection.json'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `loader_scenes` | FunctionDef | См. реализацию | [L23](../../../../scripts/cc_phone_loader_audit.py#L23) |
| `open_camera_rgb` | FunctionDef | См. реализацию | [L35](../../../../scripts/cc_phone_loader_audit.py#L35) |
| `radiometric_sample` | FunctionDef | См. реализацию | [L45](../../../../scripts/cc_phone_loader_audit.py#L45) |
| `patch_stats` | FunctionDef | Sample pixel centres in the central convex quad, using stored (x,y). | [L62](../../../../scripts/cc_phone_loader_audit.py#L62) |
| `audit` | FunctionDef | См. реализацию | [L99](../../../../scripts/cc_phone_loader_audit.py#L99) |
