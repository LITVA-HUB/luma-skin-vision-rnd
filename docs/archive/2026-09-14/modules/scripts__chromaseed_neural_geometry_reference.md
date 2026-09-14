# `scripts/chromaseed_neural_geometry_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_geometry_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent whitened-design QR/SVD and analytic perceptual metric.

SHA-256 исходника: `9ead9c956a3c89d26b84f0770d9e22e019b4ceeed1978d5a49a38cbd6fe07f2f`. Строк: **41**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from chromaseed_perceptual_reference import analytic_tensor
from scipy.linalg import qr, svdvals
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `reconstruct` | FunctionDef | См. реализацию | [L10](../../../../scripts/chromaseed_neural_geometry_reference.py#L10) |
