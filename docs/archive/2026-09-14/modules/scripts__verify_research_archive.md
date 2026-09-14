# `scripts/verify_research_archive.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/verify_research_archive.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Validate the publication against archived evidence; no data access or training.

SHA-256 исходника: `087beaa773c68fb4bc92bddfd16cc7a4cafe81aaf505837a12756de94013b692`. Строк: **77**.

## Зависимости

```python
import csv
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote
from build_research_archive import tables
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/verify_research_archive.py#L12)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/verify_research_archive.py#L15) |
