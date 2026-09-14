# `scripts/download_cc_v2_fresh.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/download_cc_v2_fresh.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Acquire the frozen INTEL-TAU 3 x 128 field-1 subset by verified ZIP ranges.

Checks only bytes, sizes, CRCs and hashes; does not decode TIFFs or read GT values.
Data license is upstream CC BY-SA 4.0, irrespective of the mirror's MIT badge.

SHA-256 исходника: `998fa1b44abb7a71bd74a4d311f7bb755192460085e7fb7437340307153de740`. Строк: **295**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/download_cc_v2_fresh.py#L20)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 21](../../../../scripts/download_cc_v2_fresh.py#L21)

```python
PROVENANCE = ROOT / "docs/data/provenance/cc_v2"
```

[Строка 22](../../../../scripts/download_cc_v2_fresh.py#L22)

```python
REVISION = "9cd6308d3cb2e383d6b185abd16fa1958bf6bc54"
```

[Строка 23](../../../../scripts/download_cc_v2_fresh.py#L23)

```python
FROZEN = {
    "Canon_5DSR_1080p.zip.field1_128.manifest.json": "0f6674ae6a27fe65cd027b7031905a8023705e4bd1bf390cc27a6dafbbc3d710",
    "Nikon_D810_1080p.zip.field1_128.manifest.json": "e0ff94721e79dd48edfe2949cdb7cb4c649024cddff5bc1c1ccfc80a1c873972",
    "Sony_IMX135_BLCCSC_1080p.zip.field1_128.manifest.json": "06488e11251c6f56ac21f45fbec931b990e7c1b10e80b8a22c65c47778b3fccc",
}
```

[Строка 28](../../../../scripts/download_cc_v2_fresh.py#L28)

```python
PAYLOAD_LIMIT = 4_500_000_000
```

[Строка 29](../../../../scripts/download_cc_v2_fresh.py#L29)

```python
TRAFFIC_LIMIT = 6_000_000_000
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `safe_destination` | FunctionDef | См. реализацию | [L32](../../../../scripts/download_cc_v2_fresh.py#L32) |
| `byte_record` | FunctionDef | См. реализацию | [L42](../../../../scripts/download_cc_v2_fresh.py#L42) |
| `unpack_member` | FunctionDef | См. реализацию | [L54](../../../../scripts/download_cc_v2_fresh.py#L54) |
| `main` | FunctionDef | См. реализацию | [L87](../../../../scripts/download_cc_v2_fresh.py#L87) |
