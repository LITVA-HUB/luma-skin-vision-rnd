# `scripts/chromaseed_palette_pretrain.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_pretrain.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed CPU auxiliary fits; no native color accuracy is inferred from this task.

SHA-256 исходника: `74e91b847187288c3761488c1bc47065b84eb2d3c8ea846ee9a5752a67613d6c`. Строк: **142**.

## Зависимости

```python
from __future__ import annotations
import json
import os
import time
import numpy as np
import torch
from chromaseed_palette_data import OUT, read, save, sha, verify
from chromaseed_palette_encoder import (
    ENCODER_PARAMETERS,
    SEEDS,
    SLOTS,
    EncoderBank,
    numpy_encode,
    transplant,
)
from chromaseed_refine import BankAdamW
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit_arrays` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_palette_pretrain.py#L23) |
| `main` | FunctionDef | См. реализацию | [L86](../../../../scripts/chromaseed_palette_pretrain.py#L86) |
