# `docs/research/cc_v6_sources/dinov2/source/dinov2/layers/attention.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/attention.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `79c7be7a452b3aad96698ec38d5d5150b9f4d8ac084fa93324510dc9f624775d`. Строк: **99**.

## Зависимости

```python
import logging
import os
import warnings
import torch
from torch import nn, Tensor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/attention.py#L21)

```python
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Attention` | nn.Module | [L36](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/attention.py#L36) |
| `MemEffAttention` | Attention | [L82](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/attention.py#L82) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Attention` | ClassDef | См. реализацию | [L36](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/attention.py#L36) |
| `MemEffAttention` | ClassDef | См. реализацию | [L82](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/attention.py#L82) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L37–55</summary>

```python
def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
```

</details>

<details><summary>forward · L69–79</summary>

```python
def forward(self, x: Tensor, is_causal: bool = False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
```

</details>

<details><summary>forward · L83–99</summary>

```python
def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

</details>
