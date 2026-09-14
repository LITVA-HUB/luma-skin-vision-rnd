# `scripts/chromaseed_architecture_scale_support_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_support_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

CPU geometry check of the provisional output-range diagnosis, not AS model acceptance.

SHA-256 исходника: `c870feeb036efde2be37a054089db30f81540e0a58ab23530052ba763cb82490`. Строк: **98**.

## Зависимости

```python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_widen_audit import direct
from chromaseed_widen_run import bank_path, check_map
from skin_local_search_train import roles, sha, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_architecture_scale_support_verify.py#L17) |
