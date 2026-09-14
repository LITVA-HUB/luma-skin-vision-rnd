# `scripts/chromaseed_gate_stability_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gate_stability_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent feature algebra, NumPy batch-one predictions and boundary checks for GS.

SHA-256 исходника: `25bd01b53125cc19e51cd3e424277b8b0b0f2af022449e64e657feeea8ded7bd`. Строк: **342**.

## Зависимости

```python
from __future__ import annotations
import argparse
import itertools
import time
from pathlib import Path
import numpy as np
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import direct_kernel, js, nz
from skin_local_search_train import CACHE_HASH, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_gate_stability_audit.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_gate_stability_audit.py#L18)

```python
ANCHORS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
```

[Строка 19](../../../../scripts/chromaseed_gate_stability_audit.py#L19)

```python
DOSES = np.array([1, 4, 16, 64], dtype=np.float64) / 255
```

[Строка 20](../../../../scripts/chromaseed_gate_stability_audit.py#L20)

```python
EPS = 1e-4
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `transformed` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_gate_stability_audit.py#L23) |
| `normalized` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_gate_stability_audit.py#L37) |
| `score` | FunctionDef | См. реализацию | [L41](../../../../scripts/chromaseed_gate_stability_audit.py#L41) |
| `metric` | FunctionDef | См. реализацию | [L48](../../../../scripts/chromaseed_gate_stability_audit.py#L48) |
| `close` | FunctionDef | См. реализацию | [L71](../../../../scripts/chromaseed_gate_stability_audit.py#L71) |
| `feasible` | FunctionDef | См. реализацию | [L86](../../../../scripts/chromaseed_gate_stability_audit.py#L86) |
| `main` | FunctionDef | См. реализацию | [L101](../../../../scripts/chromaseed_gate_stability_audit.py#L101) |
