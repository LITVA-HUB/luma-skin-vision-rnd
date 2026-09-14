# `scripts/chromaseed.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Luma ChromaSeed: synthetic color surfaces and a compact transfer model.

SHA-256 исходника: `6be6fca7d47ec25ff540119cd313c76a1ebf65e47a76ebc0a14ca8841ab175a2`. Строк: **146**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from luma_skin_vision.color import lab_to_srgb, linear_to_srgb, srgb_to_lab, srgb_to_linear
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/chromaseed.py#L10)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 14](../../../../scripts/chromaseed.py#L14)

```python
X_MEAN = np.r_[np.full(30, .5), np.zeros(6)].astype(np.float32)
```

[Строка 15](../../../../scripts/chromaseed.py#L15)

```python
X_STD = np.r_[np.full(30, .25), np.full(3, .2), np.ones(3)].astype(np.float32)
```

[Строка 16](../../../../scripts/chromaseed.py#L16)

```python
Y_MEAN = np.array([50, 0, 0], dtype=np.float32)
```

[Строка 17](../../../../scripts/chromaseed.py#L17)

```python
Y_STD = np.array([25, 30, 30], dtype=np.float32)
```

[Строка 18](../../../../scripts/chromaseed.py#L18)

```python
QUANTILES = [.01, .05, .1, .25, .5, .75, .9, .95, .99]
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `ChromaSeed` | nn.Module | [L106](../../../../scripts/chromaseed.py#L106) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `color36` | FunctionDef | Population statistics matching the real-cache feature definitions. | [L21](../../../../scripts/chromaseed.py#L21) |
| `canonical_palette` | FunctionDef | Local D65 colors, sampled without reading any real image or label. | [L41](../../../../scripts/chromaseed.py#L41) |
| `render_samples` | FunctionDef | Approximate camera/illumination randomization, not spectral skin rendering. | [L59](../../../../scripts/chromaseed.py#L59) |
| `make_palette` | FunctionDef | См. реализацию | [L91](../../../../scripts/chromaseed.py#L91) |
| `ChromaSeed` | ClassDef | 2671 learned scalars. Forward consumes normalized color36, emits normalized Lab. | [L106](../../../../scripts/chromaseed.py#L106) |
| `pack_model` | FunctionDef | См. реализацию | [L118](../../../../scripts/chromaseed.py#L118) |
| `unpack_model` | FunctionDef | См. реализацию | [L129](../../../../scripts/chromaseed.py#L129) |
| `predict` | FunctionDef | См. реализацию | [L141](../../../../scripts/chromaseed.py#L141) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>color36 · L21–38</summary>

```python
def color36(samples):
    """Population statistics matching the real-cache feature definitions."""
    z = np.asarray(samples, dtype=np.float64)
    if z.ndim != 3 or z.shape[2] != 3 or z.shape[1] < 2:
        raise ValueError("Expected batch x samples x RGB")
    if not np.isfinite(z).all() or np.any((z < 0) | (z > 1)):
        raise ValueError("Expected finite encoded RGB in [0,1]")
    mean = z.mean(1)
    centered = z - mean[:, None, :]
    std = z.std(1)
    constant = np.ptp(z, axis=1) == 0
    std[constant] = 0
    covariance = np.einsum("bpc,bpd->bcd", centered, centered) / z.shape[1]
    corr = covariance / np.maximum(std[:, :, None] * std[:, None, :], 1e-12)
    corr = np.where(constant[:, :, None] | constant[:, None, :], 0, corr)
    corr = np.clip(corr, -1, 1)
    quant = np.quantile(z, QUANTILES, axis=1).transpose(1, 0, 2).reshape(len(z), 27)
    return np.column_stack((quant, mean, std, corr[:, 0, 1], corr[:, 0, 2], corr[:, 1, 2]))
```

</details>

<details><summary>__init__ · L108–112</summary>

```python
def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(36, 64)
        self.out = nn.Linear(64, 3)
        self.skip = nn.Linear(36, 3, bias=False)
```

</details>

<details><summary>forward · L114–115</summary>

```python
def forward(self, x):
        return self.out(torch.nn.functional.silu(self.hidden(x))) + self.skip(x)
```

</details>

<details><summary>predict · L141–146</summary>

```python
def predict(model, features):
    x = (np.asarray(features, dtype=np.float32) - model["x_mean"]) / model["x_std"]
    h = x @ model["hidden_w"].T + model["hidden_b"]
    h = h / (1 + np.exp(-np.clip(h, -80, 80)))
    y = h @ model["out_w"].T + model["out_b"] + x @ model["skip_w"].T
    return y * model["y_std"] + model["y_mean"]
```

</details>
