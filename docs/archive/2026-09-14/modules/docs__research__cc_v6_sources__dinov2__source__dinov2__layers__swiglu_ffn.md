# `docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `e46d2948fb97e497cf991ff82ce30cb59b59fdf2e12a19568417113716b4f119`. Строк: **100**.

## Зависимости

```python
import os
from typing import Callable, Optional
import warnings
from torch import Tensor, nn
import torch.nn.functional as F
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 37](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py#L37)

```python
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `SwiGLUFFN` | nn.Module | [L14](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py#L14) |
| `SwiGLUFFNFused` | SwiGLU | [L54](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py#L54) |
| `SwiGLUFFNAligned` | nn.Module | [L75](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py#L75) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `SwiGLUFFN` | ClassDef | См. реализацию | [L14](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py#L14) |
| `SwiGLUFFNFused` | ClassDef | См. реализацию | [L54](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py#L54) |
| `SwiGLUFFNAligned` | ClassDef | См. реализацию | [L75](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/swiglu_ffn.py#L75) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L15–28</summary>

```python
def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
```

</details>

<details><summary>forward · L30–34</summary>

```python
def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
```

</details>

<details><summary>__init__ · L55–72</summary>

```python
def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
```

</details>

<details><summary>__init__ · L76–94</summary>

```python
def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)
```

</details>

<details><summary>forward · L96–100</summary>

```python
def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
```

</details>
