# `scripts/chromaseed_gated_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gated_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent direct-kernel/analytic-metric/SVD audit of G readouts and gates.

SHA-256 исходника: `48355e1df5f37634ded76c92be8e7b34d35772e1c7aa66a6fe5f0e3cca6b3d72`. Строк: **416**.

## Зависимости

```python
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
from chromaseed_fast_kernel import get_model
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced, reference_readouts
from chromaseed_perceptual_reference import analytic_tensor, augmented_svd
from chromaseed_refine_audit import error_summary
from scipy.linalg import lstsq
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/chromaseed_gated_audit.py#L18)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 19](../../../../scripts/chromaseed_gated_audit.py#L19)

```python
BASES = ("norm", "perceptual")
```

[Строка 20](../../../../scripts/chromaseed_gated_audit.py#L20)

```python
FAMILIES = tuple(f"{b}_{r}" for b in BASES for r in ("base", "uniform", "soft", "hard"))
```

[Строка 21](../../../../scripts/chromaseed_gated_audit.py#L21)

```python
SEEDS = (17, 29, 43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ident` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_gated_audit.py#L24) |
| `model_from` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_gated_audit.py#L30) |
| `direct_predict` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_gated_audit.py#L34) |
| `reference_gate` | FunctionDef | См. реализацию | [L52](../../../../scripts/chromaseed_gated_audit.py#L52) |
| `refit` | FunctionDef | См. реализацию | [L73](../../../../scripts/chromaseed_gated_audit.py#L73) |
| `main` | FunctionDef | См. реализацию | [L122](../../../../scripts/chromaseed_gated_audit.py#L122) |
