# `scripts/chromaseed_palette_fit_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_fit_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent parameter, optimization-diagnostic and native-encoder compatibility audit.

SHA-256 исходника: `083368029312af91a607bf6f6106f205360abc0516708cf2a3f23228ffd8a1ba`. Строк: **117**.

## Зависимости

```python
import json
import time
import numpy as np
import torch
from chromaseed_architecture_scale import Bank
from chromaseed_palette_data import OUT, read, save, sha, verify
from scipy.special import expit
from threadpoolctl import threadpool_limits
from torch.nn import functional as F
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `encoder` | FunctionDef | См. реализацию | [L14](../../../../scripts/chromaseed_palette_fit_audit.py#L14) |
| `main` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_palette_fit_audit.py#L25) |
