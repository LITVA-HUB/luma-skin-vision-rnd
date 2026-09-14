# `scripts/chromaseed_camera_support_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_camera_support_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Report the camera/support diagnostic, with all controls and matching coverage.

SHA-256 исходника: `3d697f7f3d6eacf9c7471392778e3894d02d3648c48c864ac93bd036036217a6`. Строк: **225**.

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

[Строка 17](../../../../scripts/chromaseed_camera_support_report.py#L17)

```python
VIEW_NAMES = {
    "color36": "All 36 color features",
    "rgb_mean": "Mean RGB only",
    "lab3": "Instrument Lab (oracle)",
    "lab_residual": "Color residual after Lab (oracle)",
}
```

[Строка 23](../../../../scripts/chromaseed_camera_support_report.py#L23)

```python
ROLE_NAMES = {"mixed": "Mixed cameras", "slr_to_ipod": "SLR → iPod", "ipod_to_slr": "iPod → SLR"}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_camera_support_report.py#L26) |
