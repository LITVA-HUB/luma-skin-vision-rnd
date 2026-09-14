# `scripts/skin_he_2021_acquire.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_he_2021_acquire.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Acquire the original CC BY 4.0 paired skin RGB/XYZ workbook, without editing it.

SHA-256 исходника: `0bf3f3c9c3eeabac4ea630d592bf5ec7a170bd5015077494701cb10db534da7c`. Строк: **34**.

## Зависимости

```python
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../scripts/skin_he_2021_acquire.py#L8)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 9](../../../../scripts/skin_he_2021_acquire.py#L9)

```python
PROVENANCE = ROOT / "docs/data/provenance/skin_public_2026_09_11"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
