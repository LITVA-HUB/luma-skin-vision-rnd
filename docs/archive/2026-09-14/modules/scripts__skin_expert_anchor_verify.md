# `scripts/skin_expert_anchor_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_expert_anchor_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact model/plan replay with independent observed-token provenance checks.

SHA-256 исходника: `fdba10a41aa53ecce9ab02b5f8222c0b141189c49623b1f4cb5e4836229ce9c4`. Строк: **115**.

## Зависимости

```python
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from skin_expert_anchor_train import OUT,RUN,PROTOCOL,prediction,update_digest
from skin_expert_anchor import ARMS,make_model,objective
from skin_capture_support_verify import verify_plans as original_verify_plans
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
| `verify_plans` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_expert_anchor_verify.py#L19) |
| `independent_novelty` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_expert_anchor_verify.py#L26) |
| `main` | FunctionDef | См. реализацию | [L33](../../../../scripts/skin_expert_anchor_verify.py#L33) |
