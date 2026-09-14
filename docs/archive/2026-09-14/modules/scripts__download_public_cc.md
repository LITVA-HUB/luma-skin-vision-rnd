# `scripts/download_public_cc.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/download_public_cc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Download only explicitly approved original Cube++ artifacts, with publisher MD5.

SHA-256 исходника: `d4f77b603cc4a79ffad5882ec58577a5d8346da9b67fb81e19151d08475190c5`. Строк: **69**.

## Зависимости

```python
import hashlib
import json
import time
import urllib.request
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/download_public_cc.py#L9)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 10](../../../../scripts/download_public_cc.py#L10)

```python
DATA = ROOT / "data/public/cube"
```

[Строка 11](../../../../scripts/download_public_cc.py#L11)

```python
PROVENANCE = ROOT / "docs/data/provenance/cube"
```

[Строка 12](../../../../scripts/download_public_cc.py#L12)

```python
FILES = {
    "SimpleCube++.zip": (2113441199, "d3438223e5b4ca27be874db690bef822"),
    "markup.zip": (19957846, "8c3c8344cfd16f131d0d632569d799e9"),
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L18](../../../../scripts/download_public_cc.py#L18) |
