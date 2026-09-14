# `scripts/chromaseed_fused_optimizer.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_fused_optimizer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Unintegrated fused-AdamW prototype; CPU feasibility is not GPU acceptance.

SHA-256 исходника: `71c8c14cee28317020613dff6dc016516d92e997fb89d56e18c0dc1a765c7e1c`. Строк: **76**.

## Зависимости

```python
from __future__ import annotations
import torch
from torch.optim.adamw import adamw
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `FusedBankAdamW` | — | [L9](../../../../scripts/chromaseed_fused_optimizer.py#L9) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `FusedBankAdamW` | ClassDef | Preserve one flat parameter bank while updating independent row views.  No model, sampler or backward computation changes. Floating-point update ordering differs from BankAdamW, so exact old-trajectory identity is unproven. | [L9](../../../../scripts/chromaseed_fused_optimizer.py#L9) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L16–32</summary>

```python
def __init__(self, theta, lrs, weight_decay=0.01, max_norm=5.0):
        if theta.ndim != 2 or not theta.is_contiguous() or theta.dtype != torch.float32:
            raise ValueError("contiguous FP32 slots-by-parameters bank required")
        rates = torch.as_tensor(lrs, dtype=theta.dtype, device=theta.device).clone()
        if (
            rates.shape != (theta.shape[0],)
            or not torch.isfinite(rates).all()
            or torch.any(rates <= 0)
        ):
            raise ValueError("one positive finite rate per slot required")
        if weight_decay < 0 or max_norm <= 0:
            raise ValueError("invalid decay or clipping norm")
        self.theta, self.lrs = theta, rates
        self.initial_lrs = rates.clone()
        self.m, self.v = torch.zeros_like(theta), torch.zeros_like(theta)
        self.steps = torch.zeros(theta.shape[0], dtype=torch.float32, device=theta.device)
        self.weight_decay, self.max_norm = weight_decay, max_norm
```

</details>
