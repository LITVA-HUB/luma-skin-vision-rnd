# `scripts/chromaseed_hybrid_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_hybrid_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent H geometry, selection, real NumPy queries and QR/SVD reconstructions.

SHA-256 исходника: `f29ae00d43541f4f900803e0e49001400ed49156a1380493968460bbc636adb8`. Строк: **546**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact, summaries, validate_normalizer
from chromaseed_gate_stability_audit import ANCHORS, DOSES, close, transformed
from chromaseed_gate_stability_audit import score as gate_score
from chromaseed_gated_audit import model_from
from chromaseed_hybrid_numpy import Predictor
from chromaseed_hybrid_reference import Basis, Geometry, predict
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_projection_audit import numeric_equal, scoring
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_hybrid_audit.py#L22)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 25](../../../../scripts/chromaseed_hybrid_audit.py#L25)

```python
SETTINGS = [(0.0, np.zeros(3))] + [(t, a) for t in DOSES for a in ANCHORS]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ident` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_hybrid_audit.py#L28) |
| `options` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_hybrid_audit.py#L36) |
| `ahash` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_hybrid_audit.py#L50) |
| `cached_prediction` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_hybrid_audit.py#L54) |
| `main` | FunctionDef | См. реализацию | [L96](../../../../scripts/chromaseed_hybrid_audit.py#L96) |
