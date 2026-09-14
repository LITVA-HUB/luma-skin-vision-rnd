# `scripts/chromaseed_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent alignment/aggregation audit; no model fitting or selection.

SHA-256 исходника: `a54daf8dfff8122c508e27764246550c087f19c224b2869d0a670bae7a553c07`. Строк: **139**.

## Зависимости

```python
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from chromaseed import predict
from luma_skin_vision.color import delta_e00, srgb_to_lab
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/chromaseed_audit.py#L10)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 16](../../../../scripts/chromaseed_audit.py#L16)

```python
RUN = ROOT / "experiments/runs/chromaseed_v1"
```

[Строка 17](../../../../scripts/chromaseed_audit.py#L17)

```python
FROZEN = ROOT / "experiments/runs/chromaseed_frozen_v1"
```

[Строка 18](../../../../scripts/chromaseed_audit.py#L18)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_v1"
```

[Строка 19](../../../../scripts/chromaseed_audit.py#L19)

```python
CACHE = ROOT.parents[1] / "luma-skin-vision-rnd/data/processed/skin_mskcc_pixels_v1/train.npz"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_audit.py#L22) |
| `digest` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_audit.py#L26) |
| `load` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_audit.py#L30) |
| `main` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_audit.py#L35) |
