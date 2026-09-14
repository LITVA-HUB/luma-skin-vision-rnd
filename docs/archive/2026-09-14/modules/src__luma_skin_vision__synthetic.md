# `src/luma_skin_vision/synthetic.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/synthetic.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Deterministic toy patches, never evidence for real-world facial measurement.

SHA-256 исходника: `3045798a175444c6c11ab0bb8b3b1d6c2c9c052fee6bcf6225751f15d8761179`. Строк: **80**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image
from luma_skin_vision.color import linear_to_srgb, srgb_to_lab, srgb_to_linear
from luma_skin_vision.data import Record, assign_splits, sha256
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `generate` | FunctionDef | См. реализацию | [L13](../../../../src/luma_skin_vision/synthetic.py#L13) |
