# `scripts/chromaseed_camera_support.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_camera_support.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed camera diagnostics and descriptive support; no skin-color model selection.

SHA-256 исходника: `f13cc228ef6359d269e3b794f1f78cf140f9a584f505aef482487af19da53d39`. Строк: **189**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/chromaseed_camera_support.py#L12)

```python
VIEWS = ("color36", "rgb_mean", "lab3", "lab_residual")
```

[Строка 13](../../../../scripts/chromaseed_camera_support.py#L13)

```python
METHODS = ("linear", "rbf")
```

[Строка 14](../../../../scripts/chromaseed_camera_support.py#L14)

```python
CALIPERS = (1.0, 2.0, 3.0, 5.0, 10.0)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `row_weights` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_camera_support.py#L17) |
| `aggregate_people` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_camera_support.py#L35) |
| `standardize` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_camera_support.py#L50) |
| `views` | FunctionDef | См. реализацию | [L74](../../../../scripts/chromaseed_camera_support.py#L74) |
| `kernel` | FunctionDef | См. реализацию | [L87](../../../../scripts/chromaseed_camera_support.py#L87) |
| `classify` | FunctionDef | См. реализацию | [L91](../../../../scripts/chromaseed_camera_support.py#L91) |
| `classification_metrics` | FunctionDef | См. реализацию | [L121](../../../../scripts/chromaseed_camera_support.py#L121) |
| `match_cost` | FunctionDef | См. реализацию | [L139](../../../../scripts/chromaseed_camera_support.py#L139) |
| `nearest` | FunctionDef | См. реализацию | [L153](../../../../scripts/chromaseed_camera_support.py#L153) |
| `weighted_quantile` | FunctionDef | См. реализацию | [L173](../../../../scripts/chromaseed_camera_support.py#L173) |
| `describe` | FunctionDef | См. реализацию | [L184](../../../../scripts/chromaseed_camera_support.py#L184) |
