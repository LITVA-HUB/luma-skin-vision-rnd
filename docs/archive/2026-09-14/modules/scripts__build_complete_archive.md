# `scripts/build_complete_archive.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/build_complete_archive.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Build the 2026-09-14 documentation atlas using saved text and Python AST only.

Never imports model/training modules, reads participant arrays, or runs experiments.
Run from any directory with Python >= 3.11. Existing scientific evidence is immutable.

SHA-256 исходника: `84c99b9ea80c630dcd89fe0e740d3c2b85ef60a3b1a61a6da14c1f42be998701`. Строк: **685**.

## Зависимости

```python
from __future__ import annotations
import ast
import csv
import hashlib
import json
import re
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/build_complete_archive.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/build_complete_archive.py#L17)

```python
OUT = ROOT / "docs/archive/2026-09-14"
```

[Строка 18](../../../../scripts/build_complete_archive.py#L18)

```python
PUB = ROOT / "docs/publication"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L21](../../../../scripts/build_complete_archive.py#L21) |
| `sha` | FunctionDef | См. реализацию | [L25](../../../../scripts/build_complete_archive.py#L25) |
| `write` | FunctionDef | См. реализацию | [L29](../../../../scripts/build_complete_archive.py#L29) |
| `dump` | FunctionDef | См. реализацию | [L34](../../../../scripts/build_complete_archive.py#L34) |
| `esc` | FunctionDef | См. реализацию | [L38](../../../../scripts/build_complete_archive.py#L38) |
| `rel` | FunctionDef | См. реализацию | [L42](../../../../scripts/build_complete_archive.py#L42) |
| `slug` | FunctionDef | См. реализацию | [L46](../../../../scripts/build_complete_archive.py#L46) |
| `table_extract` | FunctionDef | См. реализацию | [L50](../../../../scripts/build_complete_archive.py#L50) |
| `report_for` | FunctionDef | См. реализацию | [L66](../../../../scripts/build_complete_archive.py#L66) |
| `source_page` | FunctionDef | См. реализацию | [L85](../../../../scripts/build_complete_archive.py#L85) |
| `catalog` | FunctionDef | См. реализацию | [L272](../../../../scripts/build_complete_archive.py#L272) |
| `run_index` | FunctionDef | Index every retained run, including those without a benchmark directory. | [L480](../../../../scripts/build_complete_archive.py#L480) |
| `model_records` | FunctionDef | Literal record extraction; repeated records remain repeated evidence. | [L520](../../../../scripts/build_complete_archive.py#L520) |
| `main` | FunctionDef | См. реализацию | [L553](../../../../scripts/build_complete_archive.py#L553) |
