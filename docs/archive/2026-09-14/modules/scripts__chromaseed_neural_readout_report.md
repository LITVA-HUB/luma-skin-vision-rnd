# `scripts/chromaseed_neural_readout_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_readout_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Full NR comparison tables and model-cost report from frozen audited records.

SHA-256 исходника: `42b5fcbe2dd0932d838962adbd246b9e0164a03f5cf8e32469637b335810c5da`. Строк: **464**.

## Зависимости

```python
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from chromaseed_kernel_audit import js
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_neural_readout_report.py#L17)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 18](../../../../scripts/chromaseed_neural_readout_report.py#L18)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
```

[Строка 19](../../../../scripts/chromaseed_neural_readout_report.py#L19)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_readout_v1"
```

[Строка 20](../../../../scripts/chromaseed_neural_readout_report.py#L20)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-readout-2026-09-13.md"
```

[Строка 21](../../../../scripts/chromaseed_neural_readout_report.py#L21)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 22](../../../../scripts/chromaseed_neural_readout_report.py#L22)

```python
GROUPS = ("raw36", "mean3")
```

[Строка 23](../../../../scripts/chromaseed_neural_readout_report.py#L23)

```python
FAMILIES = ("norm", "perceptual")
```

[Строка 24](../../../../scripts/chromaseed_neural_readout_report.py#L24)

```python
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `key` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_neural_readout_report.py#L35) |
| `csv_write` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_neural_readout_report.py#L39) |
| `main` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_neural_readout_report.py#L46) |
