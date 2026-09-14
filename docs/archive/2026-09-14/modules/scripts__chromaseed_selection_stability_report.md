# `scripts/chromaseed_selection_stability_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_selection_stability_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report all fixed-OOF diagnostic groups without reading outer predictions.

SHA-256 исходника: `55ef608512ef7e23b71497d06c3d538c1cb5d4add33dea21fe7b8583a203adc2`. Строк: **122**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_selection_stability_report.py#L16)

```python
LABELS = {'norm_mse': 'Normalized ridge', 'constant_de2': 'Shared perceptual',
          'local_de2': 'Local perceptual', 'local_irls': 'Fixed-metric correction',
          'midpoint_irls': 'Midpoint correction'}
```

[Строка 19](../../../../scripts/chromaseed_selection_stability_report.py#L19)

```python
ROLE_NAMES = {'mixed': 'Mixed cameras', 'slr_to_ipod': 'SLR fit side', 'ipod_to_slr': 'iPod fit side'}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_selection_stability_report.py#L22) |
