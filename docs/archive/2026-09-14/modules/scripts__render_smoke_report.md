# `scripts/render_smoke_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/render_smoke_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Publish a LOCAL synthetic engineering report, never a real accuracy claim.

SHA-256 исходника: `ffc52148e0e5dd3f59a5dd95b35513763dd856c90c8f217c1e9d97f9bbc31b21`. Строк: **74**.

## Зависимости

```python
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from luma_skin_vision.evaluation import bootstrap_selective
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/render_smoke_report.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L17](../../../../scripts/render_smoke_report.py#L17) |
