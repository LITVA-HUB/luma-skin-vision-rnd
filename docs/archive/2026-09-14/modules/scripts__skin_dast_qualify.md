# `scripts/skin_dast_qualify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_dast_qualify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only qualification of DAST native color conventions, without inventing targets.

The numerical comparison recomputes Lab from the SAME stored XYZ with different
white references. It is neither an observer conversion nor a model-quality metric.
Original source files, existing data roles and model registrations are not edited.

SHA-256 исходника: `a7593b16ca8a93933a21087beed10bed274b9b31b9d681b174cad3eb3aa89795`. Строк: **276**.

## Зависимости

```python
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/skin_dast_qualify.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 20](../../../../scripts/skin_dast_qualify.py#L20)

```python
DATA = Path('D:/Luma-RnD/data_growth_2026_09_14')
```

[Строка 21](../../../../scripts/skin_dast_qualify.py#L21)

```python
SOURCE = DATA / 'dast_public_example'
```

[Строка 22](../../../../scripts/skin_dast_qualify.py#L22)

```python
OUT = DATA / 'dast_qualification_v1'
```

[Строка 23](../../../../scripts/skin_dast_qualify.py#L23)

```python
PALETTE = DATA / 'uminho_palette_v1/profile.json'
```

[Строка 24](../../../../scripts/skin_dast_qualify.py#L24)

```python
PROFILE_SHA = '15db398c35c9c68427f3f09a2ff9f6636c9717c37953217ec71c6637807e8dd1'
```

[Строка 25](../../../../scripts/skin_dast_qualify.py#L25)

```python
PALETTE_SHA = '008034fd2c82648fa3aa316d37c35ce3a70320681bd40810c5b62472147f97f6'
```

[Строка 26](../../../../scripts/skin_dast_qualify.py#L26)

```python
DOCUMENTS = {
    'paper.pdf': 'ebdeafa82a5e55defbc28d5fbd92ef49f637b1abbe6f63bfda71313ff125d5a1',
    'slides.pdf': 'bf2bff1f38b0e4c1547cd9789036149aa9ed2ea337afaff739d99a45c0fe8732',
    'dsm4-manual-v8.pdf': 'ff9a34011c61850585b16336fd1bd66a6b6c607edef6713189dfbc81608f6786',
    'hunterlab-whitepoints.html': '9a101785a098400b6f3d72bf7c3adcaef8489b7ab46a6506ab235269594516d1',
    'dast-readme-current.md': '5ef81b1a83615e268202949fd4758f6aa5fe77c068c06fc14e83bdaa09272252',
}
```

[Строка 33](../../../../scripts/skin_dast_qualify.py#L33)

```python
CONVENTION = dict(
    illuminant='D65', observer_degrees=10, geometry='45/0',
    instrument='Cortex DSM-4', evidence='paper.pdf p4; dsm4-manual-v8.pdf pp6-7',
    qualification_basis='documented device convention plus consistency with native Lab/XYZ',
    manual_release='2026-02-10', native_software_version='3.2.0.7',
    limit='Manual is later than the 2025 measurements; the original session configuration is not independently certified.',
    native_to_srgb_conversion_performed=False,
)
```

[Строка 41](../../../../scripts/skin_dast_qualify.py#L41)

```python
PAPER_SITES = dict(CFH='forehead', CLC='left cheek, zygomatic region',
                   CRC='right cheek, zygomatic region', CLI='left hand, interosseous region',
                   CRI='right hand, interosseous region')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read_json` | FunctionDef | См. реализацию | [L46](../../../../scripts/skin_dast_qualify.py#L46) |
| `digest` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_dast_qualify.py#L50) |
| `binding` | FunctionDef | См. реализацию | [L55](../../../../scripts/skin_dast_qualify.py#L55) |
| `triple` | FunctionDef | См. реализацию | [L60](../../../../scripts/skin_dast_qualify.py#L60) |
| `f_lab` | FunctionDef | См. реализацию | [L66](../../../../scripts/skin_dast_qualify.py#L66) |
| `f_inverse` | FunctionDef | См. реализацию | [L71](../../../../scripts/skin_dast_qualify.py#L71) |
| `xyz_to_lab` | FunctionDef | См. реализацию | [L76](../../../../scripts/skin_dast_qualify.py#L76) |
| `infer_white` | FunctionDef | См. реализацию | [L84](../../../../scripts/skin_dast_qualify.py#L84) |
| `white_diagnostics` | FunctionDef | См. реализацию | [L94](../../../../scripts/skin_dast_qualify.py#L94) |
| `qualify_profile` | FunctionDef | См. реализацию | [L112](../../../../scripts/skin_dast_qualify.py#L112) |
| `audit_originals` | FunctionDef | См. реализацию | [L167](../../../../scripts/skin_dast_qualify.py#L167) |
| `exif_diagnostics` | FunctionDef | См. реализацию | [L194](../../../../scripts/skin_dast_qualify.py#L194) |
| `prepare_result` | FunctionDef | См. реализацию | [L220](../../../../scripts/skin_dast_qualify.py#L220) |
| `main` | FunctionDef | См. реализацию | [L249](../../../../scripts/skin_dast_qualify.py#L249) |
