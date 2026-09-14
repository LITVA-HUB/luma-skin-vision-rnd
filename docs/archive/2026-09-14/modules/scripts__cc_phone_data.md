# `scripts/cc_phone_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze and sparsely acquire original Beyond RGB phone records, no pixel decoding.

SHA-256 исходника: `f53a55d2625e9a59e4ca5e68a11c5f0120e54b0b2ab322875dc8a84261215c0e`. Строк: **164**.

## Зависимости

```python
import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path
from download_cc_v2_fresh import byte_record, safe_destination, unpack_member
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/cc_phone_data.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 15](../../../../scripts/cc_phone_data.py#L15)

```python
PROVENANCE = ROOT/'docs/data/provenance/mobile_screen_2026_09_11'
```

[Строка 16](../../../../scripts/cc_phone_data.py#L16)

```python
LOCK = PROVENANCE/'beyond_rgb_selection.json'
```

[Строка 17](../../../../scripts/cc_phone_data.py#L17)

```python
PREFIX = 'beyond-unzip/beyondRGB/'
```

[Строка 18](../../../../scripts/cc_phone_data.py#L18)

```python
REQUIRED = ('CRI/CRI.txt',) + tuple(
    f'{mode}/{camera}{suffix}' for camera in ('samsung', 'oppo')
    for mode in ('NT', 'WT') for suffix in (('.h5', '_tags.json', '_cc_detection.json') if mode == 'WT' else ('.h5', '_tags.json')))
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `split_ranges` | FunctionDef | См. реализацию | [L23](../../../../scripts/cc_phone_data.py#L23) |
| `scene_members` | FunctionDef | См. реализацию | [L36](../../../../scripts/cc_phone_data.py#L36) |
| `freeze` | FunctionDef | См. реализацию | [L46](../../../../scripts/cc_phone_data.py#L46) |
| `download` | FunctionDef | См. реализацию | [L91](../../../../scripts/cc_phone_data.py#L91) |
