# `scripts/skin_spectral_palette_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_spectral_palette_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read back every palette sample and independently integrate its original spectrum.

SHA-256 исходника: `c7cc0285ac48110a9d9f0f229b999038724f7cc28915f044b609a3c1ca5eea86`. Строк: **217**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/skin_spectral_palette_audit.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 15](../../../../scripts/skin_spectral_palette_audit.py#L15)

```python
OUT = Path('D:/Luma-RnD/data_growth_2026_09_14/uminho_palette_v1')
```

[Строка 16](../../../../scripts/skin_spectral_palette_audit.py#L16)

```python
LEGACY = Path('C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/public/uminho_hsfd_v1')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_spectral_palette_audit.py#L19) |
| `read` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_spectral_palette_audit.py#L24) |
| `save` | FunctionDef | См. реализацию | [L28](../../../../scripts/skin_spectral_palette_audit.py#L28) |
| `reference_integration` | FunctionDef | Interpolate each measured spectrum directly, without production matrices. | [L34](../../../../scripts/skin_spectral_palette_audit.py#L34) |
| `erode_twice` | FunctionDef | См. реализацию | [L53](../../../../scripts/skin_spectral_palette_audit.py#L53) |
| `main` | FunctionDef | См. реализацию | [L62](../../../../scripts/skin_spectral_palette_audit.py#L62) |
