# `scripts/chromaseed_local_denoise.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_local_denoise.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Small independent continuous-Lab denoising blocks (ND).

SHA-256 исходника: `2369e7d802d7bcc8112d9deb519ac2f8a3d6c4408cb7628109c2e54a33878971`. Строк: **114**.

## Зависимости

```python
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/chromaseed_local_denoise.py#L11)

```python
SPECS = {
    "plain": (1, 36, 69),
    "local2": (2, 39, 32),
    "local4": (4, 39, 16),
    "blind4": (4, 39, 16),
    "e2e4": (4, 39, 16),
}
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Bank` | nn.Module | [L46](../../../../scripts/chromaseed_local_denoise.py#L46) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `schedule` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_local_denoise.py#L20) |
| `preprocessor` | FunctionDef | См. реализацию | [L29](../../../../scripts/chromaseed_local_denoise.py#L29) |
| `Bank` | ClassDef | Leading optimizer slots are individual blocks, never shared moments. | [L46](../../../../scripts/chromaseed_local_denoise.py#L46) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L49–68</summary>

```python
def __init__(self, family, seeds):
        super().__init__()
        if family not in SPECS or not seeds:
            raise ValueError("unknown family or empty slots")
        self.family, self.m = family, len(seeds)
        self.k, self.d, self.h = SPECS[family]
        self.p = (self.d + 1) * self.h + (self.h + 1) * 3
        initial = np.zeros((self.m, self.k, self.p), np.float32)
        for i, seed in enumerate(seeds):
            for b in range(self.k):
                rng = np.random.default_rng(int(seed) + 310003 + 1009 * b)
                initial[i, b, : self.d * self.h] = rng.uniform(
                    -1 / math.sqrt(self.d), 1 / math.sqrt(self.d), self.d * self.h
                )
                start = (self.d + 1) * self.h
                initial[i, b, start : start + self.h * 3] = rng.uniform(
                    -1 / math.sqrt(self.h), 1 / math.sqrt(self.h), self.h * 3
                )
        self.theta = nn.Parameter(torch.from_numpy(initial.reshape(self.m * self.k, self.p)))
        self.a, self.s = schedule(self.k)
```

</details>

<details><summary>export · L107–114</summary>

```python
def export(self, slot, prep):
        if not 0 <= slot < self.m:
            raise ValueError("invalid model slot")
        return {
            **{k: v.copy() for k, v in prep.items()},
            "family": np.asarray(self.family),
            "theta": self.theta.detach().reshape(self.m, self.k, self.p)[slot].cpu().numpy().copy(),
        }
```

</details>
