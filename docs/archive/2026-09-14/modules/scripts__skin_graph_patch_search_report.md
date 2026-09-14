# `scripts/skin_graph_patch_search_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_graph_patch_search_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Two independent source searches: branch combination and patch likelihood.

SHA-256 исходника: `5007e4a3b7bbce42833c0faa36498b52b48d70997fc3a48db7cd8f99d919f28a`. Строк: **123**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `comparison` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_graph_patch_search_report.py#L12) |
| `report` | FunctionDef | См. реализацию | [L21](../../../../scripts/skin_graph_patch_search_report.py#L21) |
