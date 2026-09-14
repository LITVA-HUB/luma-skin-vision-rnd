# `scripts/chromaseed_affine_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_affine_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent direct predictions, person selection and QR/SVD A refits.

SHA-256 исходника: `eedb5736f06f73d10110c93392295bd40264e8532a78c6de583ab170e495accb`. Строк: **563**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import itertools
import time
from pathlib import Path
import numpy as np
from chromaseed_affine_reference import refit
from chromaseed_gate_stability_audit import close, metric, score, transformed
from chromaseed_gated_audit import direct_predict, model_from
from chromaseed_gated_numpy import Predictor
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/chromaseed_affine_audit.py#L23)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 24](../../../../scripts/chromaseed_affine_audit.py#L24)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```

[Строка 26](../../../../scripts/chromaseed_affine_audit.py#L26)

```python
CONTROLS = ("g_norm_base", "g_norm_soft", "g_perceptual_base", "g_perceptual_soft")
```

[Строка 27](../../../../scripts/chromaseed_affine_audit.py#L27)

```python
ANCHORS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
```

[Строка 28](../../../../scripts/chromaseed_affine_audit.py#L28)

```python
DOSES = (1 / 255, 4 / 255, 16 / 255, 64 / 255)
```

[Строка 29](../../../../scripts/chromaseed_affine_audit.py#L29)

```python
SETTINGS = [(0.0, np.zeros(3), -1)] + [(t, a, j) for t in DOSES for j, a in enumerate(ANCHORS)]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ident` | FunctionDef | См. реализацию | [L32](../../../../scripts/chromaseed_affine_audit.py#L32) |
| `exact` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_affine_audit.py#L36) |
| `row_hash` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_affine_audit.py#L43) |
| `cached_prediction` | FunctionDef | См. реализацию | [L47](../../../../scripts/chromaseed_affine_audit.py#L47) |
| `scores` | FunctionDef | См. реализацию | [L60](../../../../scripts/chromaseed_affine_audit.py#L60) |
| `validate_normalizer` | FunctionDef | См. реализацию | [L70](../../../../scripts/chromaseed_affine_audit.py#L70) |
| `summaries` | FunctionDef | См. реализацию | [L79](../../../../scripts/chromaseed_affine_audit.py#L79) |
| `main` | FunctionDef | См. реализацию | [L110](../../../../scripts/chromaseed_affine_audit.py#L110) |
