# `scripts/skin_he_companion_check.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_he_companion_check.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Test proposed cross-deposit calibration correspondence on charts, not skin TEST.

SHA-256 исходника: `07c32b50c89320ed4b80127f51aca4ac8b23d594dae56513663323c26de3db95`. Строк: **41**.

## Зависимости

```python
import hashlib
import json
from pathlib import Path
import numpy as np
import openpyxl
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_he_companion_check.py#L9)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 10](../../../../scripts/skin_he_companion_check.py#L10)

```python
OUT = ROOT / "docs/data/provenance/skin_public_2026_09_11"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `cells` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_he_companion_check.py#L17) |
