# `scripts/chromaseed_selection_stability_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_selection_stability_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar/loop reconstruction of the fixed-OOF diagnostic.

SHA-256 исходника: `b3a4e147dca7531ffb2afd3910a290a03246aed1a7ebf50a213adfb3131658b2`. Строк: **164**.

## Зависимости

```python
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_selection_stability_audit.py#L14)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `js` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_selection_stability_audit.py#L17) |
| `main` | FunctionDef | См. реализацию | [L21](../../../../scripts/chromaseed_selection_stability_audit.py#L21) |
