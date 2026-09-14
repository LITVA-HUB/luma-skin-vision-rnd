# `scripts/cc_phone_resume.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_resume.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Polite transport wrapper; frozen phone selection/CRC/size logic is unchanged.

SHA-256 исходника: `149fff63f86f5a89e67df9e61754c76f43bd657ba3bd239e47e82bcbfaa467dc`. Строк: **50**.

## Зависимости

```python
import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `polite_opener` | FunctionDef | См. реализацию | [L11](../../../../scripts/cc_phone_resume.py#L11) |
