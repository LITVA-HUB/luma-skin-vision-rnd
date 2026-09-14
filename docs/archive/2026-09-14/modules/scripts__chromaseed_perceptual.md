# `scripts/chromaseed_perceptual.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_perceptual.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Compact kernel readouts with local perceptual geometry and fit-only IRLS.

SHA-256 исходника: `2c92fced648e0e72230dd38707d4b6e0e1aecf64062eb18be9c5d41e21735a2a`. Строк: **223**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_condensed_exact import condensed_width
from chromaseed_fast_kernel import ALPHAS, SEEDS, WIDTHS, _landmark_fit, payload, solve_columns
from chromaseed_kernel import EIGEN_FLOOR, coordinates, fit_normalizer
from scipy.linalg import cho_factor, cho_solve
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_perceptual.py#L14)

```python
FAMILIES = ("norm_mse", "constant_de2", "local_de2", "local_irls", "midpoint_irls")
```

[Строка 15](../../../../scripts/chromaseed_perceptual.py#L15)

```python
CHECKPOINTS = (1, 4, 16)
```

[Строка 16](../../../../scripts/chromaseed_perceptual.py#L16)

```python
RANK = 128
```

[Строка 17](../../../../scripts/chromaseed_perceptual.py#L17)

```python
TAU = 2.
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `local_tensor` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_perceptual.py#L20) |
| `coupled_ridge` | FunctionDef | См. реализацию | [L44](../../../../scripts/chromaseed_perceptual.py#L44) |
| `prepare` | FunctionDef | См. реализацию | [L64](../../../../scripts/chromaseed_perceptual.py#L64) |
| `geometry` | FunctionDef | См. реализацию | [L75](../../../../scripts/chromaseed_perceptual.py#L75) |
| `make_basis` | FunctionDef | См. реализацию | [L85](../../../../scripts/chromaseed_perceptual.py#L85) |
| `from_theta` | FunctionDef | См. реализацию | [L104](../../../../scripts/chromaseed_perceptual.py#L104) |
| `objective` | FunctionDef | См. реализацию | [L108](../../../../scripts/chromaseed_perceptual.py#L108) |
| `trajectory` | FunctionDef | См. реализацию | [L120](../../../../scripts/chromaseed_perceptual.py#L120) |
| `fit_single` | FunctionDef | См. реализацию | [L158](../../../../scripts/chromaseed_perceptual.py#L158) |
| `key` | FunctionDef | См. реализацию | [L182](../../../../scripts/chromaseed_perceptual.py#L182) |
| `fit_bank` | FunctionDef | См. реализацию | [L188](../../../../scripts/chromaseed_perceptual.py#L188) |
