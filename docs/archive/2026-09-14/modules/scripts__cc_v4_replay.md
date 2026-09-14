# `scripts/cc_v4_replay.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v4_replay.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Replay both selected checkpoints on source validation, never on test rows.

SHA-256 исходника: `189617d0924d126a252f6c7a2af7cc612adb14cd36dca057f1e74e0c65f1154d`. Строк: **72**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v4_experiment import evaluate
from cc_v4_model import CorrectionEvidenceNet
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/cc_v4_replay.py#L15)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L18](../../../../scripts/cc_v4_replay.py#L18) |
