# `scripts/skin_loss_field_palette_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_loss_field_palette_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent fit-only grid, scalar color costs and full Gram identity audit.

SHA-256 исходника: `47aa4e827ff595264c0c85756be410b254cc16bd4fdfc8625482e821b564ab9e`. Строк: **39**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from skin_loss_field_train import OUT,PRIOR
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_loss_field_palette_audit.py#L12) |
