# `scripts/verify_public_cc.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/verify_public_cc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Verify the public milestone without overwriting historical synthetic evidence.

SHA-256 исходника: `fd21b25ce32e75dbf4f85a773b6f65b8baaca0b52490068087c064e54ff25072`. Строк: **45**.

## Зависимости

```python
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from luma_skin_vision.cc.benchmark import data_hashes
from luma_skin_vision.experiment import source_identity, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
