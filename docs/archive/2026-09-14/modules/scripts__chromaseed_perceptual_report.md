# `scripts/chromaseed_perceptual_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_perceptual_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Measured P-study report and figures; no fitting or selection changes.

SHA-256 исходника: `027840194bfeb109ffa3438c7bd65ba9079cbdc402d06b13037ea73fd7263855`. Строк: **147**.

## Зависимости

```python
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_perceptual_report.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_perceptual_report.py#L17)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 18](../../../../scripts/chromaseed_perceptual_report.py#L18)

```python
FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
```

[Строка 19](../../../../scripts/chromaseed_perceptual_report.py#L19)

```python
LABELS = {"norm_mse": "Нормированная ошибка", "constant_de2": "Общие цветовые веса", "local_de2": "Веса каждого оттенка",
          "local_irls": "Коррекция с фиксированной геометрией", "midpoint_irls": "Коррекция с обновлением геометрии"}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `report` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_perceptual_report.py#L23) |
