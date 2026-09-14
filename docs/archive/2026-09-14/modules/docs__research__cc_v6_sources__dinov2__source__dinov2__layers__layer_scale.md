# `docs/research/cc_v6_sources/dinov2/source/dinov2/layers/layer_scale.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/layer_scale.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `dadd5aafe178f1bf72a205a02a6645c7e635cacbad585d4a7369c200c6e89135`. Строк: **34**.

## Зависимости

```python
from typing import Optional, Union
import torch
from torch import Tensor
from torch import nn
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `LayerScale` | nn.Module | [L15](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/layer_scale.py#L15) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `LayerScale` | ClassDef | См. реализацию | [L15](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/layer_scale.py#L15) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L16–28</summary>

```python
def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.reset_parameters()
```

</details>

<details><summary>forward · L33–34</summary>

```python
def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
```

</details>
