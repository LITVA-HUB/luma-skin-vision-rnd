# `scripts/chromaseed_fast_kernel.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_fast_kernel.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Sampled bandwidth and genuinely on-demand Nyström training, CPU float64.

SHA-256 исходника: `8d574c76c58c64be4832e8d7de7691c5b99de41ba795d6da028a9246423dc691`. Строк: **259**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import time
import numpy as np
from chromaseed_kernel import (
    EIGEN_FLOOR,
    coordinates,
    fit_normalizer,
    gaussian_kernel,
    median_width,
    nystrom_coefficients,
    select_landmarks,
)
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/chromaseed_fast_kernel.py#L18)

```python
ARMS = ("dense_exact", "column_exact", "column_pairs1024", "column_pairs4096", "column_pairs16384")
```

[Строка 19](../../../../scripts/chromaseed_fast_kernel.py#L19)

```python
RANKS = (64, 128, 256)
```

[Строка 20](../../../../scripts/chromaseed_fast_kernel.py#L20)

```python
ALPHAS = (.1, 1., 10.)
```

[Строка 21](../../../../scripts/chromaseed_fast_kernel.py#L21)

```python
WIDTHS = (.5, 1., 2.)
```

[Строка 22](../../../../scripts/chromaseed_fast_kernel.py#L22)

```python
SEEDS = (17, 29, 43)
```

[Строка 23](../../../../scripts/chromaseed_fast_kernel.py#L23)

```python
PAIR_BUDGETS = (1024, 4096, 16384)
```

[Строка 24](../../../../scripts/chromaseed_fast_kernel.py#L24)

```python
PAIR_SEED_OFFSET = 104729
```

[Строка 25](../../../../scripts/chromaseed_fast_kernel.py#L25)

```python
FIELDS = ("x_mean", "x_std", "y_mean", "y_std", "width", "centers", "coefficient")
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `KernelColumns` | — | [L74](../../../../scripts/chromaseed_fast_kernel.py#L74) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `pair_indices` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_fast_kernel.py#L28) |
| `_positive_median` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_fast_kernel.py#L36) |
| `_sampled_width_bank` | FunctionDef | См. реализацию | [L44](../../../../scripts/chromaseed_fast_kernel.py#L44) |
| `sampled_width` | FunctionDef | См. реализацию | [L70](../../../../scripts/chromaseed_fast_kernel.py#L70) |
| `KernelColumns` | ClassDef | No NxN matrix; count actual queried entries instead of inferred FLOPs. | [L74](../../../../scripts/chromaseed_fast_kernel.py#L74) |
| `streaming_landmarks` | FunctionDef | См. реализацию | [L97](../../../../scripts/chromaseed_fast_kernel.py#L97) |
| `solve_columns` | FunctionDef | См. реализацию | [L130](../../../../scripts/chromaseed_fast_kernel.py#L130) |
| `model_id` | FunctionDef | См. реализацию | [L153](../../../../scripts/chromaseed_fast_kernel.py#L153) |
| `payload` | FunctionDef | См. реализацию | [L157](../../../../scripts/chromaseed_fast_kernel.py#L157) |
| `_landmark_fit` | FunctionDef | См. реализацию | [L162](../../../../scripts/chromaseed_fast_kernel.py#L162) |
| `fit_one` | FunctionDef | См. реализацию | [L175](../../../../scripts/chromaseed_fast_kernel.py#L175) |
| `fit_bank` | FunctionDef | См. реализацию | [L198](../../../../scripts/chromaseed_fast_kernel.py#L198) |
| `flatten_bank` | FunctionDef | См. реализацию | [L238](../../../../scripts/chromaseed_fast_kernel.py#L238) |
| `get_model` | FunctionDef | См. реализацию | [L242](../../../../scripts/chromaseed_fast_kernel.py#L242) |
| `evaluate_bank` | FunctionDef | См. реализацию | [L246](../../../../scripts/chromaseed_fast_kernel.py#L246) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L77–85</summary>

```python
def __init__(self, x, width):
        self.x = np.asarray(x, np.float64)
        if self.x.ndim != 2 or not len(self.x) or self.x.shape[1] != 36 or not np.isfinite(self.x).all():
            raise ValueError("finite color36 required")
        if not np.isfinite(width) or width <= 0:
            raise ValueError("positive finite width required")
        self.squared_norm = np.einsum("ij,ij->i", self.x, self.x)
        self.denominator = 2. * self.x.shape[1] * float(width) ** 2
        self.entries_evaluated = 0
```

</details>
