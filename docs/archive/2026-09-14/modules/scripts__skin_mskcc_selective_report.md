# `scripts/skin_mskcc_selective_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_selective_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Render frozen test aggregates; no fitting, ranking selection or image output.

SHA-256 исходника: `805f7b952b426167c698262aefac65e36a93974bbc8940134c44f9bbe63a267d`. Строк: **78**.

## Зависимости

```python
import csv,json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_mskcc_data import ROOT,sha
from skin_mskcc_selective_core import OUT,verify_lock
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L11](../../../../scripts/skin_mskcc_selective_report.py#L11) |
