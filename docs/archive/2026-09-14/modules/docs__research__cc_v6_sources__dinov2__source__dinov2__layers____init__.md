# `docs/research/cc_v6_sources/dinov2/source/dinov2/layers/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/layers/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `1b55deed39d5ab0b589bef421bbfe06f24a10cd590a0f8403234c9ee4e34109d`. Строк: **12**.

## Зависимости

```python
from .dino_head import DINOHead
from .layer_scale import LayerScale
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused, SwiGLUFFNAligned
from .block import NestedTensorBlock, CausalAttentionBlock
from .attention import Attention, MemEffAttention
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
