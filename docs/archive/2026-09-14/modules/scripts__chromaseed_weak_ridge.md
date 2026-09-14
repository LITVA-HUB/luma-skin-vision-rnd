# `scripts/chromaseed_weak_ridge.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_weak_ridge.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Expanded positive ridge grid; numerical operations reuse frozen P helpers.

SHA-256 исходника: `32662cb75298e84ab5aeee76e3efd2965d779c67c385c2992d2e41a0c6e6b6ca`. Строк: **85**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_perceptual import (
    CHECKPOINTS,
    FAMILIES,
    RANK,
    SEEDS,
    WIDTHS,
    coupled_ridge,
    from_theta,
    geometry,
    key,
    make_basis,
    prepare,
    trajectory,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_weak_ridge.py#L22)

```python
ALPHAS = (.0001, .0003, .001, .003, .01, .03, .1, 1., 10.)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit_single` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_weak_ridge.py#L25) |
| `fit_bank` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_weak_ridge.py#L49) |
