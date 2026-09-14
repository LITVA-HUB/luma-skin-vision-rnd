# `scripts/skin_expert_anchor_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_expert_anchor_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Aggregate actual skin color accuracy and explicitly labeled mode diagnostics.

SHA-256 исходника: `cdf6e1ba0a2b28e956c710fef1427dc0bb060b8b8cdd4cf1a9e5b05376d1dcdf`. Строк: **118**.

## Зависимости

```python
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skin_expert_anchor_train import OUT,RUN
from skin_expert_anchor import ARMS
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write,subset
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_expert_anchor_report.py#L17) |
