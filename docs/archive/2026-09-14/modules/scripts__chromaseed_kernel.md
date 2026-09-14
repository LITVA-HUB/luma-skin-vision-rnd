# `scripts/chromaseed_kernel.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_kernel.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Compact RKHS predictors; FP32 storage and explicit FP64 kernel arithmetic.

SHA-256 исходника: `65433ab44c0758606947ded87ae809f7520d12c644726fa62044f54cefef55bf`. Строк: **237**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../scripts/chromaseed_kernel.py#L7)

```python
EIGEN_FLOOR = 1e-8
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `AdaptiveKernel` | — | [L188](../../../../scripts/chromaseed_kernel.py#L188) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `gaussian_kernel` | FunctionDef | См. реализацию | [L10](../../../../scripts/chromaseed_kernel.py#L10) |
| `fit_normalizer` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_kernel.py#L22) |
| `coordinates` | FunctionDef | См. реализацию | [L32](../../../../scripts/chromaseed_kernel.py#L32) |
| `median_width` | FunctionDef | См. реализацию | [L36](../../../../scripts/chromaseed_kernel.py#L36) |
| `_eigh` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_kernel.py#L50) |
| `exact_coefficients` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_kernel.py#L61) |
| `select_landmarks` | FunctionDef | Nested pivot selection on sqrt(W) K sqrt(W), without target access. | [L74](../../../../scripts/chromaseed_kernel.py#L74) |
| `landmark_whitener` | FunctionDef | См. реализацию | [L115](../../../../scripts/chromaseed_kernel.py#L115) |
| `nystrom_coefficients` | FunctionDef | См. реализацию | [L126](../../../../scripts/chromaseed_kernel.py#L126) |
| `projection_coefficients` | FunctionDef | См. реализацию | [L143](../../../../scripts/chromaseed_kernel.py#L143) |
| `residual_diagnostic` | FunctionDef | RKHS norm of the ACTUAL (possibly rounded) coefficient difference. | [L152](../../../../scripts/chromaseed_kernel.py#L152) |
| `residual_bound` | FunctionDef | Per-channel bound against the teacher, NOT against instrument truth. | [L164](../../../../scripts/chromaseed_kernel.py#L164) |
| `predict_kernel` | FunctionDef | См. реализацию | [L174](../../../../scripts/chromaseed_kernel.py#L174) |
| `pack_adaptive` | FunctionDef | См. реализацию | [L179](../../../../scripts/chromaseed_kernel.py#L179) |
| `AdaptiveKernel` | ClassDef | См. реализацию | [L188](../../../../scripts/chromaseed_kernel.py#L188) |
| `convex_blend` | FunctionDef | См. реализацию | [L234](../../../../scripts/chromaseed_kernel.py#L234) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L189–204</summary>

```python
def __init__(self, payload):
        self.prep = payload
        self.centers = payload["centers"].astype(np.float64)
        self.width = float(payload["width"])
        self.ranks = payload["ranks"].astype(int)
        if np.any(np.diff(self.ranks) <= 0) or self.ranks[-1] != len(self.centers):
            raise ValueError("strictly nested center counts ending at payload size required")
        self.tolerance = float(payload["tolerance"])
        self.levels = []
        start = 0
        for i, rank in enumerate(self.ranks):
            self.levels.append((payload["coefficients"][start:start + rank].astype(np.float64),
                                payload["norm_sq"][i], payload["residual_values"][start:start + rank]))
            start += rank
        if start != len(payload["coefficients"]) or start != len(payload["residual_values"]):
            raise ValueError("invalid concatenated prefix payload")
```

</details>
