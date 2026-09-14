# `scripts/chromaseed_affine_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_affine_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent augmented-design QR/SVD reference for A; no primary Gram solver.

SHA-256 исходника: `4e67dc2b7452b08188268ea11ce8189ea80503d211eaa0a7724994dfb92aee24`. Строк: **93**.

## Зависимости

```python
from __future__ import annotations
import itertools
import numpy as np
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import reference_gate
from chromaseed_kernel_audit import direct_kernel, norm
from chromaseed_perceptual_audit import balanced
from chromaseed_perceptual_reference import analytic_tensor
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `qr_ridge` | FunctionDef | См. реализацию | [L15](../../../../scripts/chromaseed_affine_reference.py#L15) |
| `refit` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_affine_reference.py#L34) |
