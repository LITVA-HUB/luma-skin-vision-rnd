# `src/luma_skin_vision/cc/v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/cc/v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched direct and anchor-relative color-constancy models, without novelty claims.

For positive diagonal gains D and anchors above EPS, a(Dx)=D a(x).
The anchored encoder consequently sees the same x/a(x), and its illuminant
transforms equivariantly, up to L2 normalization. This algebra is not evidence
that real camera/ISP changes are diagonal, nor a guarantee of physical accuracy.

SHA-256 исходника: `205eba622f6c9707df0c947e8db06cec0708f257c24dffba64385fcb5a071307`. Строк: **159**.

## Зависимости

```python
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../src/luma_skin_vision/cc/v2.py#L16)

```python
EPS = 1e-8
```

[Строка 17](../../../../src/luma_skin_vision/cc/v2.py#L17)

```python
CHANNELS = ("r", "g", "b")
```

[Строка 18](../../../../src/luma_skin_vision/cc/v2.py#L18)

```python
CHEAP_FEATURE_COLUMNS = (
    [
        f"log_pred_over_{anchor}_{channel}_minus_green"
        for anchor in ("gw", "sog", "max")
        for channel in ("r", "b")
    ]
    + [
        f"patch2x2_{row}{col}_{channel}_over_global_gw"
        for channel in CHANNELS
        for row in range(2)
        for col in range(2)
    ]
    + [f"spatial_std_{channel}_over_global_gw" for channel in CHANNELS]
)
```

[Строка 32](../../../../src/luma_skin_vision/cc/v2.py#L32)

```python
FEATURE_UNITS = "dimensionless; natural log ratios for first six cheap columns"
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `CompactResidualCC` | nn.Module | [L73](../../../../src/luma_skin_vision/cc/v2.py#L73) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `validate_image` | FunctionDef | No input clipping: reject unsupported pixels before computing an anchor. | [L35](../../../../src/luma_skin_vision/cc/v2.py#L35) |
| `_anchor` | FunctionDef | См. реализацию | [L45](../../../../src/luma_skin_vision/cc/v2.py#L45) |
| `channel_anchor` | FunctionDef | Unnormalized channelwise Minkowski mean, including masked zero pixels.  Returns Nx3. The positive numerical floor breaks exact homogeneity only near degenerate channels. Such samples are flagged invalid by input_validity(). | [L55](../../../../src/luma_skin_vision/cc/v2.py#L55) |
| `input_validity` | FunctionDef | Each channel must have a mean above EPS; this is not a quality certificate. | [L67](../../../../src/luma_skin_vision/cc/v2.py#L67) |
| `CompactResidualCC` | ClassDef | См. реализацию | [L73](../../../../src/luma_skin_vision/cc/v2.py#L73) |
| `gain_augment` | FunctionDef | Transformed-real augmentation under a diagonal model, not new physical GT.  No post-gain clipping is applied. Metadata makes the exact transformation auditable; zero magnitude still applies common exposure and random flips. | [L104](../../../../src/luma_skin_vision/cc/v2.py#L104) |
| `risk_features_invariant` | FunctionDef | Relative thumbnail features; no absolute RGB/prediction or camera metadata.  Cheap columns are invariant when pred transforms equivariantly and channels stay above the floor. Context is invariant only for the anchored models. The direct mode has no diagonal-invariance promise. Invalid rows are finite fallbacks and must be rejected regardless of any learned risk score. | [L131](../../../../src/luma_skin_vision/cc/v2.py#L131) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L74–87</summary>

```python
def __init__(self, mode="direct", backbone="small"):
        super().__init__()
        if mode not in ("direct", "gw", "sog") or backbone not in ("small", "large"):
            raise ValueError("Unsupported mode or backbone")
        self.mode = mode
        self.backbone_name = backbone
        factory, width = (
            (mobilenet_v3_small, 576) if backbone == "small" else (mobilenet_v3_large, 960)
        )
        self.backbone = factory(weights=None).features
        self.context = nn.Sequential(nn.Linear(width, 64), nn.SiLU())
        self.illuminant = nn.Linear(64, 3)
        nn.init.zeros_(self.illuminant.weight)
        nn.init.zeros_(self.illuminant.bias)
```

</details>

<details><summary>forward · L89–101</summary>

```python
def forward(self, image):
        validate_image(image)
        # Branches depend only on immutable architecture, never on pixel values.
        if self.mode == "direct":
            rms = image.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(EPS**2).sqrt()
            encoder_input = image / rms
            anchor = torch.ones_like(image[:, :, 0, 0])
        else:
            anchor = _anchor(image, 1 if self.mode == "gw" else 6)
            encoder_input = torch.log1p((image / anchor[:, :, None, None]).clamp(max=8))
        context = self.context(self.backbone(encoder_input).mean(dim=(-2, -1)))
        residual = self.illuminant(context).clamp(-2, 2).exp()
        return F.normalize(anchor * residual, dim=-1), context
```

</details>
