# `scripts/skin_distribution_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_distribution_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact replay plus scalar skin-error and manual quadrature decision audit.

SHA-256 исходника: `de9ad4642cf21027ee61e43fd7f0a1a427fcf6609b9b554eff6eeea1ff68c13f`. Строк: **103**.

## Зависимости

```python
import argparse,itertools,json,math
from pathlib import Path
import numpy as np
import torch
from skin_distribution_train import OUT,RUN,PROTOCOL,prediction,endpoints
from skin_distribution_model import ColorDistribution,ARMS
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `manual_decision` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_distribution_audit.py#L15) |
| `main` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_distribution_audit.py#L36) |
