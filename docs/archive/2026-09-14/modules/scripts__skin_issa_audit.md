# `scripts/skin_issa_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_issa_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Audit native formula/cache consistency on explicitly enabled ISSA roles.

SHA-256 исходника: `3ed9b900967267fbec80952826a01ed2dab31f5ee6160d47af49970a5f65d892`. Строк: **101**.

## Зависимости

```python
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_issa_data import ROOT, OUT, PRIVATE, RAW, NS, constants, endpoint_rows, sha, write_json, column_name
from skin_issa_color import spectral_xyz, source_lab
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `run` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_issa_audit.py#L17) |
