# `scripts/chromaseed_palette_encoder.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_encoder.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Shared AS-compatible local encoder and normalization-preserving transfer.

SHA-256 исходника: `7f538734867adf56fb0feade0501e5c955be24a50978ad3894a6c0b434990991`. Строк: **79**.

## Зависимости

```python
from __future__ import annotations
import math
import zlib
import numpy as np
import torch
from chromaseed_refine import _Layer
from scipy.special import expit
from torch import nn
from torch.nn import functional as F
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_palette_encoder.py#L14)

```python
SEEDS = (17, 29, 43)
```

[Строка 15](../../../../scripts/chromaseed_palette_encoder.py#L15)

```python
SLOTS = tuple((s, a) for s in SEEDS for a in ('aligned', 'shuffled'))
```

[Строка 16](../../../../scripts/chromaseed_palette_encoder.py#L16)

```python
ENCODER_PARAMETERS = 105856
```

[Строка 17](../../../../scripts/chromaseed_palette_encoder.py#L17)

```python
SPECS = (('token1', 18, 384), ('token2', 384, 256), ('palette_aux', 256, 36))
```

[Строка 18](../../../../scripts/chromaseed_palette_encoder.py#L18)

```python
TOTAL_PARAMETERS = 115108
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `EncoderBank` | nn.Module | [L21](../../../../scripts/chromaseed_palette_encoder.py#L21) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `EncoderBank` | ClassDef | См. реализацию | [L21](../../../../scripts/chromaseed_palette_encoder.py#L21) |
| `numpy_encode` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_palette_encoder.py#L49) |
| `transplant` | FunctionDef | См. реализацию | [L65](../../../../scripts/chromaseed_palette_encoder.py#L65) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L22–35</summary>

```python
def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(6, TOTAL_PARAMETERS))
        self.layers = {}
        offset = 0
        with torch.no_grad():
            for name, a, b in SPECS:
                self.layers[name] = _Layer(self, offset, a, b)
                size = (a+1)*b
                for i, (seed, _) in enumerate(SLOTS):
                    g = torch.Generator().manual_seed(seed+zlib.crc32(name.encode()))
                    self.theta[i, offset:offset+size].uniform_(-1/math.sqrt(a), 1/math.sqrt(a), generator=g)
                offset += size
        assert offset == TOTAL_PARAMETERS
```

</details>

<details><summary>forward · L40–41</summary>

```python
def forward(self, x):
        return self.layers['palette_aux'](self.encode(x))
```

</details>

<details><summary>export · L43–46</summary>

```python
def export(self, slot, mean, std):
        return dict(theta=self.theta[slot, :ENCODER_PARAMETERS].detach().numpy().copy(),
                    mean=np.asarray(mean, np.float32).copy(), std=np.asarray(std, np.float32).copy(),
                    seed=np.asarray(SLOTS[slot][0]), arm=np.asarray(SLOTS[slot][1]))
```

</details>
