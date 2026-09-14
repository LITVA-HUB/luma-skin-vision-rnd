# `scripts/skin_graph_support_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_graph_support_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact model/plan replay with independent observed-token provenance checks.

SHA-256 исходника: `80d792d0b68a76d7a0c403b85032d48e77640fba8a5db3872b0fd7d1b424c966`. Строк: **103**.

## Зависимости

```python
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_graph_support_train import OUT,RUN,PROTOCOL,prediction,update_digest
from skin_graph_support_model import GraphSupportColor,ARMS
from skin_capture_support_verify import verify_plans as original_verify_plans
from skin_expert_anchor_verify import independent_novelty
from skin_capture_support import make_plan,apply_plan
from skin_capture_model import CaptureColor,MODES
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `verify_plans` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_graph_support_verify.py#L20) |
| `main` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_graph_support_verify.py#L27) |
