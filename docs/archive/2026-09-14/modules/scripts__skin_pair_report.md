# `scripts/skin_pair_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_pair_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate completed source experiments; cannot train or change selection.

SHA-256 исходника: `d0a2fe00be9df1e340ea7bdfa22a785287c0723516aaee12b3991da582b6428f`. Строк: **61**.

## Зависимости

```python
import json
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_pair_train import OUT,SEEDS
from skin_pair_invariance import ARMS
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L9](../../../../scripts/skin_pair_report.py#L9) |
