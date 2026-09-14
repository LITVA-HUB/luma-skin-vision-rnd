# `scripts/chromaseed_selection_stability_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_selection_stability_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Registered diagnostic: no model fitting, no outer prediction/result access.

SHA-256 исходника: `8897b564e6d1963a8d4c7ef36161e1c2669a695c54e9516d902344b469624e10`. Строк: **140**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from chromaseed_selection_stability import bootstrap_counts, diagnose, person_losses
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_selection_stability_run.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 15](../../../../scripts/chromaseed_selection_stability_run.py#L15)

```python
REPEATS = 20_000
```

[Строка 16](../../../../scripts/chromaseed_selection_stability_run.py#L16)

```python
BOOT_SEEDS = (2026091301, 2026091302, 2026091303)
```

[Строка 17](../../../../scripts/chromaseed_selection_stability_run.py#L17)

```python
MODEL_SEEDS = (17, 29, 43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_selection_stability_run.py#L20) |
| `lock_inputs` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_selection_stability_run.py#L24) |
| `name_for` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_selection_stability_run.py#L54) |
| `main` | FunctionDef | См. реализацию | [L59](../../../../scripts/chromaseed_selection_stability_run.py#L59) |
