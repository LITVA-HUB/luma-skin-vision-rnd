# `scripts/skin_mskcc_ablation_report_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_ablation_report_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent replay and scalar color audit for the12adaptive source ablations.

SHA-256 исходника: `acac0e18be6993b0d6490bd093aa745a9b3eb96548f2661fafcd7c204bf3a6f9`. Строк: **65**.

## Зависимости

```python
import json
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load,PROTOCOL
from skin_mskcc_vote_v2 import GlobalColorMLP,VoteAblation
from skin_mskcc_train_ablation_v2 import predict,ABLATION_PROTOCOL
from skin_mskcc_audit import scalar_de
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_mskcc_ablation_report_v2.py#L14) |
