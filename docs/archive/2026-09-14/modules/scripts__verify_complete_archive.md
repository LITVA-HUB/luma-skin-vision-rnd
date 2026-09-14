# `scripts/verify_complete_archive.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/verify_complete_archive.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Verify the publication without importing or running any research implementation.

SHA-256 исходника: `8f4a0ad30223350ebbbfa952f345a833241e91cf61e69ddfe70e95f5b902dbef`. Строк: **170**.

## Зависимости

```python
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/verify_complete_archive.py#L11)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 12](../../../../scripts/verify_complete_archive.py#L12)

```python
OUT = ROOT / "docs/archive/2026-09-14"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `load` | FunctionDef | См. реализацию | [L15](../../../../scripts/verify_complete_archive.py#L15) |
| `sha` | FunctionDef | См. реализацию | [L19](../../../../scripts/verify_complete_archive.py#L19) |
| `main` | FunctionDef | См. реализацию | [L23](../../../../scripts/verify_complete_archive.py#L23) |
