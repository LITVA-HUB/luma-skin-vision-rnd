# `scripts/verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Record local verification evidence; exit nonzero on any failed check.

SHA-256 исходника: `b8316f473aac40bb1e532e14bacdefdb1eb776b3108a31e3bc3428eefeb1782a`. Строк: **66**.

## Зависимости

```python
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from luma_skin_vision.experiment import source_identity
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L14](../../../../scripts/verify.py#L14) |
