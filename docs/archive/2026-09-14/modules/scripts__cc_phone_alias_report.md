# `scripts/cc_phone_alias_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_alias_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent FP64 phone rescoring and transparent report generation.

SHA-256 исходника: `ce192a94b35694b0da82654f578a86543c1e6972eb44db54a7db32bc520af2e2`. Строк: **106**.

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

[Строка 14](../../../../scripts/cc_phone_alias_report.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 15](../../../../scripts/cc_phone_alias_report.py#L15)

```python
B = ROOT / "docs/benchmarks/phone_v1_alias"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `aggregate` | FunctionDef | См. реализацию | [L46](../../../../scripts/cc_phone_alias_report.py#L46) |
