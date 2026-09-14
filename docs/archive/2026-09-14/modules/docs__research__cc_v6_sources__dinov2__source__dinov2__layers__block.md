# `docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `60c0ac7dfa4474be313fabfa5a23d82faf6f0cecd4e720a88be35de9788cb636`. Строк: **316**.

## Зависимости

```python
import logging
import os
from typing import Callable, List, Any, Tuple, Dict, Optional
import warnings
import torch
from torch import nn, Tensor
from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 27](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L27)

```python
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Block` | nn.Module | [L43](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L43) |
| `CausalAttentionBlock` | nn.Module | [L117](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L117) |
| `NestedTensorBlock` | Block | [L267](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L267) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Block` | ClassDef | См. реализацию | [L43](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L43) |
| `CausalAttentionBlock` | ClassDef | См. реализацию | [L117](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L117) |
| `drop_add_residual_stochastic_depth` | FunctionDef | См. реализацию | [L173](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L173) |
| `get_branges_scales` | FunctionDef | См. реализацию | [L197](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L197) |
| `add_residual` | FunctionDef | См. реализацию | [L205](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L205) |
| `get_attn_bias_and_cat` | FunctionDef | this will perform the index select, cat the tensors, and provide the attn_bias from cache | [L220](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L220) |
| `drop_add_residual_stochastic_depth_list` | FunctionDef | См. реализацию | [L244](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L244) |
| `NestedTensorBlock` | ClassDef | См. реализацию | [L267](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/block.py#L267) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L44–87</summary>

```python
def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path
```

</details>

<details><summary>forward · L89–114</summary>

```python
def forward(self, x: Tensor) -> Tensor:
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x
```

</details>

<details><summary>__init__ · L118–146</summary>

```python
def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = Attention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
```

</details>

<details><summary>forward · L164–170</summary>

```python
def forward(
        self,
        x: torch.Tensor,
    ):
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn
```

</details>

<details><summary>forward · L308–316</summary>

```python
def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError
```

</details>
