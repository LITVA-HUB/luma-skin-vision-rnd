# `scripts/cc_v5_replay.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v5_replay.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

CPU replay of completed V5 checkpoints; verify saved GPU predictions.

SHA-256 исходника: `10ff62f31e077d98ff078ef1471e822f9da5ac4d3560bcf33dbe18b2ac975cb1`. Строк: **81**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v4_experiment import evaluate
from cc_v5_model import CorrectionEvidenceNet
from cc_v5_report import ARMS
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/cc_v5_replay.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/cc_v5_replay.py#L19) |
