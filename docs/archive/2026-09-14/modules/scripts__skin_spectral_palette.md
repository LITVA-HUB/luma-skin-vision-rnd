# `scripts/skin_spectral_palette.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_spectral_palette.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Measured TRAIN spectra with Seg1 sampling and explicitly derived D65 colors.

SHA-256 исходника: `0a46160ea32a6f65baac251ec102ab3547492b3f005826166f21fed6d5b8c93f`. Строк: **327**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/skin_spectral_palette.py#L15)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 16](../../../../scripts/skin_spectral_palette.py#L16)

```python
DATA = Path('D:/Luma-RnD/data_growth_2026_09_14')
```

[Строка 17](../../../../scripts/skin_spectral_palette.py#L17)

```python
OUT = DATA / 'uminho_palette_v1'
```

[Строка 18](../../../../scripts/skin_spectral_palette.py#L18)

```python
LEGACY = Path('C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/public/uminho_hsfd_v1')
```

[Строка 19](../../../../scripts/skin_spectral_palette.py#L19)

```python
PROFILE = DATA / 'uminho_train_v2/profile.json'
```

[Строка 20](../../../../scripts/skin_spectral_palette.py#L20)

```python
PROFILE_SHA = '1fc862ebaeb65ae46528fb9a4ad33fe90f90f858dab6bfc355388abf40b4e226'
```

[Строка 21](../../../../scripts/skin_spectral_palette.py#L21)

```python
MANIFEST_SHA = '3d3b7dc7256eb04250dee2f5ef76880f4101d019313d97ba051480f8d320b5c7'
```

[Строка 22](../../../../scripts/skin_spectral_palette.py#L22)

```python
CHECKPOINT = DATA / 'facial_skin_v1/best.pt'
```

[Строка 23](../../../../scripts/skin_spectral_palette.py#L23)

```python
CHECKPOINT_SHA = '1203cbc5ed2ee17cb2a408c47a23b3a28468f169d48b4d3cee8bd3059e7fbed3'
```

[Строка 24](../../../../scripts/skin_spectral_palette.py#L24)

```python
PROTOCOL = ROOT / 'docs/research/skin_spectral_palette_v1_protocol.md'
```

[Строка 25](../../../../scripts/skin_spectral_palette.py#L25)

```python
CIE2 = ROOT / 'docs/data/provenance/skin_public_2026_09_11'
```

[Строка 26](../../../../scripts/skin_spectral_palette.py#L26)

```python
CIE10 = ROOT / 'docs/data/provenance/cie_material_v1'
```

[Строка 27](../../../../scripts/skin_spectral_palette.py#L27)

```python
CIE_FILES = [
    (CIE2 / 'CIE_xyz_1931_2deg.csv', CIE2 / 'CIE_xyz_1931_2deg.csv_metadata.json'),
    (CIE10 / 'CIE_xyz_1964_10deg.csv', CIE10 / 'CIE_xyz_1964_10deg.csv_metadata.json'),
    (CIE10 / 'CIE_std_illum_D65.csv', CIE10 / 'CIE_std_illum_D65.csv_metadata_v2.json'),
]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_spectral_palette.py#L34) |
| `read_json` | FunctionDef | См. реализацию | [L39](../../../../scripts/skin_spectral_palette.py#L39) |
| `save_json` | FunctionDef | См. реализацию | [L43](../../../../scripts/skin_spectral_palette.py#L43) |
| `integration_matrix` | FunctionDef | См. реализацию | [L49](../../../../scripts/skin_spectral_palette.py#L49) |
| `xyz_lab` | FunctionDef | См. реализацию | [L66](../../../../scripts/skin_spectral_palette.py#L66) |
| `sample_indices` | FunctionDef | См. реализацию | [L78](../../../../scripts/skin_spectral_palette.py#L78) |
| `qualified_mask` | FunctionDef | См. реализацию | [L89](../../../../scripts/skin_spectral_palette.py#L89) |
| `checked_profile` | FunctionDef | См. реализацию | [L104](../../../../scripts/skin_spectral_palette.py#L104) |
| `original_cie` | FunctionDef | См. реализацию | [L121](../../../../scripts/skin_spectral_palette.py#L121) |
| `freeze` | FunctionDef | См. реализацию | [L139](../../../../scripts/skin_spectral_palette.py#L139) |
| `verify_lock` | FunctionDef | См. реализацию | [L161](../../../../scripts/skin_spectral_palette.py#L161) |
| `preview_tile` | FunctionDef | См. реализацию | [L172](../../../../scripts/skin_spectral_palette.py#L172) |
| `prepare` | FunctionDef | См. реализацию | [L186](../../../../scripts/skin_spectral_palette.py#L186) |
