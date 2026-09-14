# `docs/research/cc_v6_sources/dinov2/source/dinov2/layers/patch_embed.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/patch_embed.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `40da6add3d811198ea3e17cb99cdd4e5cda59e369efbbe3d18d89308618cf142`. Строк: **88**.

## Зависимости

```python
from typing import Callable, Optional, Tuple, Union
from torch import Tensor
import torch.nn as nn
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `PatchEmbed` | nn.Module | [L25](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/patch_embed.py#L25) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `make_2tuple` | FunctionDef | См. реализацию | [L16](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/patch_embed.py#L16) |
| `PatchEmbed` | ClassDef | 2D image to patch embedding: (B,C,H,W) -> (B,N,D)  Args:     img_size: Image size.     patch_size: Patch token size.     in_chans: Number of input image channels.     embed_dim: Number of linear projection output channels.     norm_layer: Normalization layer. | [L25](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/patch_embed.py#L25) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L37–66</summary>

```python
def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
```

</details>

<details><summary>forward · L68–81</summary>

```python
def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x
```

</details>
