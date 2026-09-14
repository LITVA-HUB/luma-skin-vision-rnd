# `scripts/chromaseed_crossfit_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_crossfit_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

C whole-person routing, independent teacher/head refits and actual consumer audit.

SHA-256 исходника: `c5f725573f42458d99f61a3f55428386a65cf7ac268975d2ab76a3d28d891855`. Строк: **448**.

## Зависимости

```python
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact, summaries, validate_normalizer
from chromaseed_crossfit_reference import head, teacher
from chromaseed_gate_stability_audit import ANCHORS, DOSES, close, transformed
from chromaseed_gate_stability_audit import score as gate_score
from chromaseed_gated_audit import model_from
from chromaseed_hybrid_numpy import Predictor
from chromaseed_hybrid_reference import Basis, Geometry, predict
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/chromaseed_crossfit_audit.py#L21)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 22](../../../../scripts/chromaseed_crossfit_audit.py#L22)

```python
PARENT = ROOT / "experiments/runs/chromaseed_hybrid_v1"
```

[Строка 24](../../../../scripts/chromaseed_crossfit_audit.py#L24)

```python
TRANSFORMS = [(0.0, np.zeros(3))] + [(t, a) for t in DOSES for a in ANCHORS]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_crossfit_audit.py#L27) |
