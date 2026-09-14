# `scripts/run_public_cc_replications.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/run_public_cc_replications.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed repetitions and held-out-camera protocol; sequential to avoid GPU contention.

SHA-256 исходника: `1371b439dc621827f24d00e2b0bd4ba45017d87efa0153d8c9c27cc8d52c81e5`. Строк: **51**.

## Зависимости

```python
import json
import os
import subprocess
import sys
from pathlib import Path
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
