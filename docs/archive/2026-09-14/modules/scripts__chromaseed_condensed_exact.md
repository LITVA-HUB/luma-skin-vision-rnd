# `scripts/chromaseed_condensed_exact.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_condensed_exact.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact lower-median width via compiled condensed distances, no sampling.

SHA-256 исходника: `81b9483e2f744eee6dd43e195023d9c98a4587bd5e1157caf1ca66aa4bd858ec`. Строк: **45**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_fast_kernel import ALPHAS, WIDTHS, _landmark_fit, payload, solve_columns
from chromaseed_kernel import coordinates, fit_normalizer
from scipy.spatial.distance import pdist
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `condensed_width` | FunctionDef | См. реализацию | [L12](../../../../scripts/chromaseed_condensed_exact.py#L12) |
| `fit_condensed` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_condensed_exact.py#L33) |
