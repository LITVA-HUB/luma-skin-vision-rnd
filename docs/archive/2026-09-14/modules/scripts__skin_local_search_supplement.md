# `scripts/skin_local_search_supplement.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_search_supplement.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Publish aggregate ablations and a standalone scientific comparison figure.

SHA-256 исходника: `b7847d838b965ae217713fd5c4d797343cf3e6c2fe0b0183f30783990ec84890`. Строк: **93**.

## Зависимости

```python
import json
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/skin_local_search_supplement.py#L11)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 12](../../../../scripts/skin_local_search_supplement.py#L12)

```python
RUN = ROOT / "experiments/runs/skin_local_search_v1"
```

[Строка 13](../../../../scripts/skin_local_search_supplement.py#L13)

```python
OUT = ROOT / "docs/benchmarks/skin_local_search_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_local_search_supplement.py#L16) |
| `main` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_local_search_supplement.py#L20) |
