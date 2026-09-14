# `scripts/chromaseed_selection_stability.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_selection_stability.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Sensitivity of model selection conditional on fixed person-held-out predictions.

SHA-256 исходника: `f17987aa5f424d2c2d3c42da398573589739278a8e5999e458831f69547664c9`. Строк: **100**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `person_losses` | FunctionDef | См. реализацию | [L9](../../../../scripts/chromaseed_selection_stability.py#L9) |
| `bootstrap_counts` | FunctionDef | См. реализацию | [L19](../../../../scripts/chromaseed_selection_stability.py#L19) |
| `tie_order` | FunctionDef | См. реализацию | [L33](../../../../scripts/chromaseed_selection_stability.py#L33) |
| `winner_indices` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_selection_stability.py#L38) |
| `original_ranks` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_selection_stability.py#L43) |
| `distribution` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_selection_stability.py#L49) |
| `paired_range` | FunctionDef | См. реализацию | [L55](../../../../scripts/chromaseed_selection_stability.py#L55) |
| `diagnose` | FunctionDef | См. реализацию | [L60](../../../../scripts/chromaseed_selection_stability.py#L60) |
