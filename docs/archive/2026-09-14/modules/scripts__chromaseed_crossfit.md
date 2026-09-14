# `scripts/chromaseed_crossfit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_crossfit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed-H compact corrections with matched/included and excluded-person targets.

SHA-256 исходника: `bdb97d79ff1e773a30dc57637a8893d85ddd320c2b92b189f86dc09969002394`. Строк: **219**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_affine import color_metrics, design_matrix, gram_stats, solve_stats
from chromaseed_gated import fit_gate
from chromaseed_hybrid import package, predict, projection_fit, raw_fit, shared_basis, support
from chromaseed_kernel import gaussian_kernel
from chromaseed_perceptual import prepare
from chromaseed_projection import export
from skin_local_search_train import weights_for
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `routing` | FunctionDef | См. реализацию | [L19](../../../../scripts/chromaseed_crossfit.py#L19) |
| `route` | FunctionDef | См. реализацию | [L40](../../../../scripts/chromaseed_crossfit.py#L40) |
| `teacher_fit` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_crossfit.py#L58) |
| `teacher_pool` | FunctionDef | См. реализацию | [L91](../../../../scripts/chromaseed_crossfit.py#L91) |
| `students` | FunctionDef | См. реализацию | [L125](../../../../scripts/chromaseed_crossfit.py#L125) |
| `fit_bank` | FunctionDef | См. реализацию | [L180](../../../../scripts/chromaseed_crossfit.py#L180) |
| `fit_single` | FunctionDef | См. реализацию | [L192](../../../../scripts/chromaseed_crossfit.py#L192) |
