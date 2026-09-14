# `scripts/skin_local_search_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_search_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate local frozen experiment receipts without participant identifiers.

SHA-256 исходника: `cfe921278e038f520bc4288f5f1bd39df770b16fc7f6724c02a3ff7fd6f332e4`. Строк: **64**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../scripts/skin_local_search_report.py#L7)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 8](../../../../scripts/skin_local_search_report.py#L8)

```python
RUN = ROOT / "experiments/runs/skin_local_search_v1"
```

[Строка 9](../../../../scripts/skin_local_search_report.py#L9)

```python
OUT = ROOT / "docs/benchmarks/skin_local_search_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_local_search_report.py#L12) |
