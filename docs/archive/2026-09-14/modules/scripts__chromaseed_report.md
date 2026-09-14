# `scripts/chromaseed_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate ChromaSeed transfer evidence, retaining controls and learning curves.

SHA-256 исходника: `2ddf96ea301d023e384e8e339acc99c60d215c34767183fde1292c81a3badecb`. Строк: **154**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/chromaseed_report.py#L12)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 13](../../../../scripts/chromaseed_report.py#L13)

```python
RUN = ROOT / "experiments/runs/chromaseed_v1"
```

[Строка 14](../../../../scripts/chromaseed_report.py#L14)

```python
FROZEN = ROOT / "experiments/runs/chromaseed_frozen_v1"
```

[Строка 15](../../../../scripts/chromaseed_report.py#L15)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_v1"
```

[Строка 16](../../../../scripts/chromaseed_report.py#L16)

```python
PROTOCOLS = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 17](../../../../scripts/chromaseed_report.py#L17)

```python
ARMS = ("scratch", "clean_palette", "rendered_palette", "shuffled_palette", "skin_long")
```

[Строка 18](../../../../scripts/chromaseed_report.py#L18)

```python
LABELS = {"scratch": "С нуля", "clean_palette": "Чистая палитра", "rendered_palette": "Палитра + камера",
          "shuffled_palette": "Перемешанные ответы", "skin_long": "Дольше на коже", "random_basis": "Случайная основа"}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_report.py#L22) |
| `aggregate` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_report.py#L26) |
| `main` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_report.py#L46) |
