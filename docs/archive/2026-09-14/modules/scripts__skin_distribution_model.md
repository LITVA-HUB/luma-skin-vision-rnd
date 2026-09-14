# `scripts/skin_distribution_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_distribution_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Compact native-Lab density and deterministic downstream color decisions.

Original local implementation of established mixture-density/Bayes principles.
Quadrature and finite candidate decisions are approximations, not error bounds.

SHA-256 исходника: `5fe8a50df5d287c91e54b8b00195558764219f44e698390c62ce481d882c9414`. Строк: **77**.

## Зависимости

```python
import itertools
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skin_capture_model import CaptureColor
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/skin_distribution_model.py#L15)

```python
ARMS=['mse_mode','mse','gaussian','mdn4']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `ColorDistribution` | CaptureColor | [L18](../../../../scripts/skin_distribution_model.py#L18) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ColorDistribution` | ClassDef | См. реализацию | [L18](../../../../scripts/skin_distribution_model.py#L18) |
| `density_nll` | FunctionDef | См. реализацию | [L37](../../../../scripts/skin_distribution_model.py#L37) |
| `quadrature` | FunctionDef | Tensor Gauss-Hermite rule for normal components; deterministic FP64. | [L42](../../../../scripts/skin_distribution_model.py#L42) |
| `color_decision` | FunctionDef | Choose among mean, component centers and +/-0.25,0.5 marginal SD offsets.  No reference target is accepted. Expected errors integrate the predicted distribution only, and require independent calibration before deployment. | [L59](../../../../scripts/skin_distribution_model.py#L59) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L19–23</summary>

```python
def __init__(self):
        super().__init__('mixture')
        self.scale_head=nn.Linear(512,12)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias,math.log(math.expm1(.95)))
```

</details>

<details><summary>forward · L25–34</summary>

```python
def forward(self,x):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1))
        weights=v[...,12].softmax(1)
        means=(v[...,:12].reshape(len(x),h.shape[1],4,3)*weights[:,:,None,None]).sum(1)
        logits=self.gate(context)
        point=(means*logits.softmax(1)[...,None]).sum(1)
        scales=.05+F.softplus(self.scale_head(context).reshape(len(x),4,3))
        return point,logits,means,scales
```

</details>
