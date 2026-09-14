# `scripts/skin_train_branch_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_train_branch_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

All follow-up fits plus strong compatible historical source controls.

SHA-256 исходника: `dcb32eeba697bf15ab877846ea0bbb916eb2a0c00959f272bb2cdbe477a09233`. Строк: **87**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_train_branch_train import OUT,RUN
from skin_train_branch_model import ARMS
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `summarize_rows` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_train_branch_report.py#L10) |
| `main` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_train_branch_report.py#L18) |
