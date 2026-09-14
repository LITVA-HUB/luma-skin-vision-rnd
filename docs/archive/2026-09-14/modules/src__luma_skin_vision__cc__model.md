# `src/luma_skin_vision/cc/model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/cc/model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Standard MobileNetV3-small regressor and matched hypothesis-mixture variant.

SHA-256 исходника: `33529c224373bd99db5cf69186b235c67c5249023a1e5e07192d7387d4894005`. Строк: **36**.

## Зависимости

```python
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `CompactCC` | nn.Module | [L9](../../../../src/luma_skin_vision/cc/model.py#L9) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `CompactCC` | ClassDef | См. реализацию | [L9](../../../../src/luma_skin_vision/cc/model.py#L9) |
| `reproduction_loss` | FunctionDef | См. реализацию | [L32](../../../../src/luma_skin_vision/cc/model.py#L32) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L10–17</summary>

```python
def __init__(self, mixture=False):
        super().__init__()
        self.mixture = mixture
        self.backbone = mobilenet_v3_small(weights=None).features
        self.context = nn.Sequential(nn.Linear(576, 64), nn.SiLU())
        self.illuminant = nn.Linear(64, 3)
        # Same parameter budget in baseline; unused mixture branch excluded in forward.
        self.weights = nn.Sequential(nn.Linear(64 + 12, 32), nn.SiLU(), nn.Linear(32, 5))
```

</details>

<details><summary>forward · L19–29</summary>

```python
def forward(self, image, experts):
        pooled = self.backbone(image).mean(dim=(-1, -2))
        context = self.context(pooled)
        direct = F.normalize(F.softplus(self.illuminant(context)) + 1e-5, dim=-1)
        logits = self.weights(torch.cat([context, experts.flatten(1)], dim=1))
        if self.mixture:
            candidates = torch.cat([experts, direct[:, None]], dim=1)
            pred = F.normalize((logits.softmax(-1)[..., None] * candidates).sum(1), dim=-1)
        else:
            pred = direct
        return pred, context
```

</details>
