# `scripts/skin_he_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_he_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read the untouched author workbook into role-separated numerical intermediates.

SHA-256 исходника: `dd8646dda08ecafc448d9664779a119aab7073fc2bf63dfbfa13fb094a779813`. Строк: **72**.

## Зависимости

```python
import argparse
import hashlib
import json
import math
from pathlib import Path
import openpyxl
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/skin_he_data.py#L10)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 11](../../../../scripts/skin_he_data.py#L11)

```python
DATA = ROOT / "data/processed/skin_he_xyz_v1"
```

[Строка 12](../../../../scripts/skin_he_data.py#L12)

```python
BENCH = ROOT / "docs/benchmarks/skin_he_xyz_v1"
```

[Строка 13](../../../../scripts/skin_he_data.py#L13)

```python
WORKBOOK_SHA = "e3ad5b30a828c542ba2d23dd7fe57cddf0c3de2a163f1815cd2e482a816484a8"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_he_data.py#L16) |
| `prepare` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_he_data.py#L20) |
