# `scripts/chromaseed_external_capacity.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_external_capacity.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Count architecture storage only; no training, images or weight downloads.

SHA-256 исходника: `99472729a926ee1a1a105c415a0ebbb581dbb9abb6417284a8817db746e10ab0`. Строк: **62**.

## Зависимости

```python
import argparse
import hashlib
import json
from pathlib import Path
import torch
import torchvision
from torch import nn
from torchvision.models import efficientnet_b0, efficientnet_b4
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `inspect_capacity` | FunctionDef | См. реализацию | [L14](../../../../scripts/chromaseed_external_capacity.py#L14) |
