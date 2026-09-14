# `scripts/chromaseed_projection_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent X geometry, controls, selection, portable queries and refits.

SHA-256 исходника: `25a29bfac35d580d452eb210a23d10921518a677a3c2d4257ec9f8afcded5897`. Строк: **626**.

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
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_projection_numpy import Predictor
from chromaseed_projection_reference import (
    direct_width,
    latent,
    predict,
    projector,
    refit,
    spectrum,
)
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 32](../../../../scripts/chromaseed_projection_audit.py#L32)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 33](../../../../scripts/chromaseed_projection_audit.py#L33)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```

[Строка 35](../../../../scripts/chromaseed_projection_audit.py#L35)

```python
REPS = [("raw", 36, None)] + [
    (f"d{d}_t{label}", d, t)
    for d in (8, 16, 36)
    for label, t in (("0", 0.0), ("01", 0.1), ("05", 0.5), ("1", 1.0))
]
```

[Строка 40](../../../../scripts/chromaseed_projection_audit.py#L40)

```python
CONTROLS = (
    "g_norm_soft",
    "g_perceptual_soft",
    "a_norm_joint_guarded",
    "a_perceptual_joint_guarded",
)
```

[Строка 46](../../../../scripts/chromaseed_projection_audit.py#L46)

```python
SETTINGS = [(0.0, np.zeros(3))] + [(t, a) for t in DOSES for a in ANCHORS]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ident` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_projection_audit.py#L49) |
| `ahash` | FunctionDef | См. реализацию | [L53](../../../../scripts/chromaseed_projection_audit.py#L53) |
| `scoring` | FunctionDef | См. реализацию | [L57](../../../../scripts/chromaseed_projection_audit.py#L57) |
| `cached_output` | FunctionDef | См. реализацию | [L65](../../../../scripts/chromaseed_projection_audit.py#L65) |
| `numeric_equal` | FunctionDef | См. реализацию | [L78](../../../../scripts/chromaseed_projection_audit.py#L78) |
| `main` | FunctionDef | См. реализацию | [L87](../../../../scripts/chromaseed_projection_audit.py#L87) |
