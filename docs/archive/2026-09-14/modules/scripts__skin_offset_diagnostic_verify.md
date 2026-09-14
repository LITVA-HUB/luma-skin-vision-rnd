# `scripts/skin_offset_diagnostic_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_offset_diagnostic_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Explicit exclusion matrix, scalar color replay and residual energy identity.

SHA-256 исходника: `6ce98da3fb7fe9886aa287485760f12f33603fdd307642ff1e2a74b1179c884b`. Строк: **68**.

## Зависимости

```python
import json,math
from pathlib import Path
import numpy as np
from skin_offset_diagnostic import OUT,RUN,SOURCE_RUN,bindings,excluded_person_offsets
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_support_curve_verify import check_summary
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_offset_diagnostic_verify.py#L12) |
