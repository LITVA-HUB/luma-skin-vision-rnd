# `src/luma_skin_vision/roi/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/roi/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Coarse cheek geometry and deterministic measurement suitability proxies.

Bounding-box cheeks are provisional. They are not learned skin segmentation.
No skin-color threshold is used to exclude darker skin by assumption.

SHA-256 исходника: `2d67d724f485bef06f5ced169f7d0f6d129ff4fe7704c8810a8d10a6f9e8307e`. Строк: **61**.

## Зависимости

```python
import numpy as np
from luma_skin_vision.color import delta_e00, srgb_to_lab
from luma_skin_vision.photometry import hypotheses
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `cheek_masks` | FunctionDef | См. реализацию | [L13](../../../../src/luma_skin_vision/roi/__init__.py#L13) |
| `usable_mask` | FunctionDef | См. реализацию | [L25](../../../../src/luma_skin_vision/roi/__init__.py#L25) |
| `robust_rgb` | FunctionDef | См. реализацию | [L29](../../../../src/luma_skin_vision/roi/__init__.py#L29) |
| `measure` | FunctionDef | См. реализацию | [L38](../../../../src/luma_skin_vision/roi/__init__.py#L38) |
| `ambiguity_features` | FunctionDef | См. реализацию | [L42](../../../../src/luma_skin_vision/roi/__init__.py#L42) |
