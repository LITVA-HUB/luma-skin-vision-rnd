# `scripts/skin_local_reference_transfer_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_reference_transfer_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent source-bank exclusion, weighted solves and color-risk replay.

SHA-256 исходника: `151f4f57c6859ac95c69ee78eef6c919e036779ef5582b45e216a87845f033f0`. Строк: **55**.

## Зависимости

```python
import json,math
from pathlib import Path
import numpy as np
from skin_local_reference_transfer import OUT,RUN,bindings
from skin_local_reference_verify import verify_bank,verify_metrics
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_local_reference_transfer_verify.py#L13) |
