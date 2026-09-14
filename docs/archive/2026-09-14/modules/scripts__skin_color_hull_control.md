# `scripts/skin_color_hull_control.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_color_hull_control.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Same empirical color-support control; projection does not use image truth.

SHA-256 исходника: `f0070395a85aeba1ca3ec28cc5e3746acedf0ca3d8aa53cfb2566b8c58b0a281`. Строк: **29**.

## Зависимости

```python
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize,nnls
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `make_hull` | FunctionDef | См. реализацию | [L7](../../../../scripts/skin_color_hull_control.py#L7) |
| `project_hull` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_color_hull_control.py#L13) |
