# `scripts/cc_v2_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v2_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate every frozen evaluation; never select a model from target errors.

SHA-256 исходника: `ef79cfd17bca7f68847cdeb4d52e046061a5fce29d2ad12858577e4f4e82fdb8`. Строк: **117**.

## Зависимости

```python
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `collect` | FunctionDef | См. реализацию | [L16](../../../../scripts/cc_v2_report.py#L16) |
