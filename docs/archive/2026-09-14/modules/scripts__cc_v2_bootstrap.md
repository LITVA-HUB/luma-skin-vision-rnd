# `scripts/cc_v2_bootstrap.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v2_bootstrap.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Paired dataset uncertainty conditional on frozen trained seeds and selectors.

SHA-256 исходника: `3258887e924c78b1a0d9096ab065ded5dd250e1ffae7e867aeae04cbd9ef7d17`. Строк: **128**.

## Зависимости

```python
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `records` | FunctionDef | См. реализацию | [L15](../../../../scripts/cc_v2_bootstrap.py#L15) |
| `compare` | FunctionDef | См. реализацию | [L26](../../../../scripts/cc_v2_bootstrap.py#L26) |
