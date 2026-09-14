# `scripts/skin_shared_bias_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_shared_bias_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate all predeclared shared-bias fits, preserving ordinary comparators.

SHA-256 исходника: `ae168a75c642bb978e9dfe8d30ecdc2c3cf3ae23c390d23a8041f6ad741ca0e6`. Строк: **108**.

## Зависимости

```python
import csv
import json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT, sha
from skin_shared_bias_train import OUT, RUN
from skin_shared_bias import ARMS
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L11](../../../../scripts/skin_shared_bias_report.py#L11) |
