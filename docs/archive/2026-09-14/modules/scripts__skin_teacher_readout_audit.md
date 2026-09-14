# `scripts/skin_teacher_readout_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_teacher_readout_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent weighted-equation, scaler, metric and source-split audit.

SHA-256 исходника: `6ec6f77b4576bff0c8cfa2ae8d5054a9dfdaa13f534c2ca912cec3c0446b4298`. Строк: **100**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_mskcc_data import ROOT, sha
from skin_mskcc_audit import scalar_de
from skin_teacher_readout import OUT, RUN, ARMS, ALPHAS, source_data
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L11](../../../../scripts/skin_teacher_readout_audit.py#L11) |
