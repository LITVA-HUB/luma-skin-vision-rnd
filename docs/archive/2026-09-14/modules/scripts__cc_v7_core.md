# `scripts/cc_v7_core.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_core.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Compact student and physically limited sensor-response augmentation.

SHA-256 исходника: `6177930800baae05a93d7c146651a12353250c65b8b36ee80800acb19d9c8313`. Строк: **59**.

## Зависимости

```python
import torch
from torch import nn
from torch.nn import functional as F
from luma_skin_vision.cc.v2 import EPS, CompactResidualCC, input_validity, validate_image
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `SemanticColorNet` | CompactResidualCC | [L46](../../../../scripts/cc_v7_core.py#L46) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sensor_matrix` | FunctionDef | См. реализацию | [L9](../../../../scripts/cc_v7_core.py#L9) |
| `sensor_transform` | FunctionDef | См. реализацию | [L23](../../../../scripts/cc_v7_core.py#L23) |
| `teacher_render` | FunctionDef | См. реализацию | [L31](../../../../scripts/cc_v7_core.py#L31) |
| `SemanticColorNet` | ClassDef | См. реализацию | [L46](../../../../scripts/cc_v7_core.py#L46) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L47–49</summary>

```python
def __init__(self):
        super().__init__("direct","large")
        self.teacher_projection=nn.Conv2d(960,384,1)
```

</details>

<details><summary>forward · L51–59</summary>

```python
def forward(self,image):
        validate_image(image)
        rms=image.square().mean((1,2,3),keepdim=True).clamp_min(EPS**2).sqrt()
        features=self.backbone(image/rms)
        context=self.context(features.mean((-2,-1)))
        pred=F.normalize(self.illuminant(context).clamp(-2,2).exp(),dim=-1)
        pooled=features if features.shape[-2:]==(4,4) else F.adaptive_avg_pool2d(features,(4,4))
        projected=self.teacher_projection(pooled).flatten(2).transpose(1,2)
        return {"pred":pred,"context":context,"teacher_features":F.normalize(projected,dim=-1),"valid":input_validity(image)}
```

</details>
