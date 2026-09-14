# `scripts/cc_phone_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent FP64 phone rescoring and transparent report generation.

SHA-256 исходника: `528e8fb4bccd70eeccc1db1b49ff8f58ef9313cd11a9ceb7cec54c370cff6c84`. Строк: **93**.

## Зависимости

```python
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/cc_phone_report.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 15](../../../../scripts/cc_phone_report.py#L15)

```python
B = ROOT / "docs/benchmarks/phone_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `aggregate` | FunctionDef | См. реализацию | [L46](../../../../scripts/cc_phone_report.py#L46) |
