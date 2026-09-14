# `scripts/chromaseed_gate_stability_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gate_stability_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

All-dose GS report; synthetic boundary sensitivity does not measure failure frequency.

SHA-256 исходника: `7fc599f68a59d605b37fa3d9951a752aac30524861d04de1920607811f36a6c9`. Строк: **313**.

## Зависимости

```python
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js, nz
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_gate_stability_report.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_gate_stability_report.py#L18)

```python
ROLES = {"mixed": "Mixed", "slr_to_ipod": "SLR → iPod", "ipod_to_slr": "iPod → SLR"}
```

[Строка 19](../../../../scripts/chromaseed_gate_stability_report.py#L19)

```python
FAMILIES = tuple(
    f"{b}_{r}" for b in ("norm", "perceptual") for r in ("base", "uniform", "soft", "hard")
)
```

[Строка 22](../../../../scripts/chromaseed_gate_stability_report.py#L22)

```python
DOSES = np.array([1, 4, 16, 64]) / 255
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_gate_stability_report.py#L25) |
