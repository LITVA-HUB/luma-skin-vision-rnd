# `src/luma_skin_vision/models/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/models/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Same MobileNetV3-small backbone across C, C+ and proposed infrastructure.

C+/proposed attention is weakly supervised through color loss, not a validated
measurement mask. No semantic skin labels or downloaded pretrained weights.

SHA-256 исходника: `146f642fbeb95723ce9ea37eba9763c30d50f38cac20508148cc63480810a99a`. Строк: **50**.

## Зависимости

```python
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `ColorRegressor` | nn.Module | [L12](../../../../src/luma_skin_vision/models/__init__.py#L12) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ColorRegressor` | ClassDef | См. реализацию | [L12](../../../../src/luma_skin_vision/models/__init__.py#L12) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L13–37</summary>

```python
def __init__(self, method="baseline_c", target_mean=None, target_scale=None):
        super().__init__()
        self.method = method
        self.backbone = mobilenet_v3_small(weights=None).features
        self.attention = nn.Conv2d(576, 1, 1) if method != "baseline_c" else None
        self.aux_dim = 12 if method == "proposed_v1" else (6 if method == "baseline_c_plus" else 0)
        self.head = nn.Sequential(
            nn.Linear(576 + self.aux_dim, 128), nn.Hardswish(), nn.Linear(128, 3)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer(
            "target_mean",
            torch.as_tensor(
                [50.0, 0.0, 0.0] if target_mean is None else target_mean, dtype=torch.float32
            ),
        )
        self.register_buffer(
            "target_scale",
            torch.as_tensor(
                [20.0, 20.0, 20.0] if target_scale is None else target_scale, dtype=torch.float32
            ),
        )
        self.register_buffer("aux_mean", torch.zeros(12))
        self.register_buffer("aux_scale", torch.ones(12))
```

</details>

<details><summary>forward · L39–50</summary>

```python
def forward(self, image, ambiguity):
        features = self.backbone(image)
        if self.attention is None:
            pooled = features.mean(dim=(2, 3))
        else:
            weights = torch.softmax(self.attention(features).flatten(2), dim=-1)
            pooled = (features.flatten(2) * weights).sum(dim=-1)
        if self.aux_dim:
            aux = (ambiguity - self.aux_mean) / self.aux_scale
            aux = aux if self.aux_dim == 12 else aux[:, 6:]
            pooled = torch.cat([pooled, aux], dim=1)
        return self.head(pooled) * self.target_scale + self.target_mean
```

</details>
