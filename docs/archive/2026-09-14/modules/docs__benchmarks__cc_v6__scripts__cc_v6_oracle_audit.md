# `docs/benchmarks/cc_v6/scripts/cc_v6_oracle_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v6/scripts/cc_v6_oracle_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Post-hoc correction of unused V6 oracle diagnostics; training outputs stay immutable.

SHA-256 исходника: `dd05baa0585c610027000b65bb05046e0b515ada6c5ee9901f9b4757c8cf0a8c`. Строк: **63**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import source_rows
from cc_v4_experiment import action_rgb
from luma_skin_vision.cc.core import reproduction
from luma_skin_vision.cc.v2 import channel_anchor
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `corrected_oracle` | FunctionDef | См. реализацию | [L18](../../../../docs/benchmarks/cc_v6/scripts/cc_v6_oracle_audit.py#L18) |
| `run` | FunctionDef | См. реализацию | [L33](../../../../docs/benchmarks/cc_v6/scripts/cc_v6_oracle_audit.py#L33) |
