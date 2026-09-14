# `scripts/chromaseed_kernel_bank.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_kernel_bank.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Reusable algebra banks for ChromaSeed-K; fit and query labels are separate.

SHA-256 исходника: `67d30b3c457135620a0b8ca245512b36c05082d77e0eb71b514af8f89d4feca9`. Строк: **161**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_kernel import (
    coordinates,
    exact_coefficients,
    fit_normalizer,
    gaussian_kernel,
    median_width,
    nystrom_coefficients,
    pack_adaptive,
    projection_coefficients,
    residual_bound,
    residual_diagnostic,
    select_landmarks,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_kernel_bank.py#L22)

```python
ALPHAS = (.1, 1., 10.)
```

[Строка 23](../../../../scripts/chromaseed_kernel_bank.py#L23)

```python
WIDTHS = (.5, 1., 2.)
```

[Строка 24](../../../../scripts/chromaseed_kernel_bank.py#L24)

```python
RANKS = (16, 32, 64, 128)
```

[Строка 25](../../../../scripts/chromaseed_kernel_bank.py#L25)

```python
SEEDS = (17, 29, 43)
```

[Строка 26](../../../../scripts/chromaseed_kernel_bank.py#L26)

```python
FAMILIES = ("exact", "nys_random", "nys_pivot", "nys_rpchol", "project_rpchol")
```

[Строка 27](../../../../scripts/chromaseed_kernel_bank.py#L27)

```python
MODEL_FIELDS = ("x_mean", "x_std", "y_mean", "y_std", "width", "centers", "coefficient")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `seeds_for` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_kernel_bank.py#L30) |
| `model_id` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_kernel_bank.py#L34) |
| `fit_bank` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_kernel_bank.py#L38) |
| `flatten_bank` | FunctionDef | См. реализацию | [L93](../../../../scripts/chromaseed_kernel_bank.py#L93) |
| `get_model` | FunctionDef | См. реализацию | [L104](../../../../scripts/chromaseed_kernel_bank.py#L104) |
| `evaluate_bank` | FunctionDef | No query labels: one query-kernel per width is reused over every readout. | [L108](../../../../scripts/chromaseed_kernel_bank.py#L108) |
| `make_adaptive` | FunctionDef | См. реализацию | [L136](../../../../scripts/chromaseed_kernel_bank.py#L136) |
| `choose_from_trace` | FunctionDef | См. реализацию | [L152](../../../../scripts/chromaseed_kernel_bank.py#L152) |
