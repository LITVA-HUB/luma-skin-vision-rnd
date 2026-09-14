# `scripts/skin_local_reference_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_reference_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent weighted augmented solves and scalar skin metrics.

SHA-256 исходника: `5f56a92eaa54a6094e73276e2fd737ea9ceda57ea0298c01828cc2bb2be3b9ae`. Строк: **99**.

## Зависимости

```python
import json,math
from pathlib import Path
import numpy as np
from skin_local_reference_run import OUT,RUN,PREVIOUS,bindings
from skin_local_reference import predict_bank
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `weighted_solve` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_local_reference_verify.py#L13) |
| `verify_bank` | FunctionDef | См. реализацию | [L21](../../../../scripts/skin_local_reference_verify.py#L21) |
| `verify_metrics` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_local_reference_verify.py#L44) |
| `main` | FunctionDef | См. реализацию | [L52](../../../../scripts/skin_local_reference_verify.py#L52) |
