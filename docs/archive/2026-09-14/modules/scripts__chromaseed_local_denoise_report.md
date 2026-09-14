# `scripts/chromaseed_local_denoise_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_local_denoise_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Publish local ND evidence and seal it; subsequent calls verify read-only.

SHA-256 исходника: `8dd3e3dfb6e5c9edb496341a6e02c1800c33c5a2a23af2f14f6398f46fffe2d2`. Строк: **331**.

## Зависимости

```python
from __future__ import annotations
import csv
import json
import re
import subprocess
import sys
from collections import Counter
import numpy as np
from chromaseed_kernel_audit import js
from chromaseed_local_denoise_train import ROOT, RUN
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_local_denoise_report.py#L17)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_local_denoise_v1"
```

[Строка 18](../../../../scripts/chromaseed_local_denoise_report.py#L18)

```python
CARD = ROOT / "docs/architecture/chromaseed_local_denoise_model_card.md"
```

[Строка 19](../../../../scripts/chromaseed_local_denoise_report.py#L19)

```python
NEXT = ROOT / "docs/research/chromaseed_local_denoise_next_decision.md"
```

[Строка 20](../../../../scripts/chromaseed_local_denoise_report.py#L20)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-local-denoise-2026-09-13.md"
```

[Строка 21](../../../../scripts/chromaseed_local_denoise_report.py#L21)

```python
FAMILIES = ("plain", "local2", "local4", "blind4", "e2e4", "fg_norm_static", "random_head")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `check_map` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_local_denoise_report.py#L24) |
| `csv_out` | FunctionDef | См. реализацию | [L29](../../../../scripts/chromaseed_local_denoise_report.py#L29) |
| `main` | FunctionDef | См. реализацию | [L44](../../../../scripts/chromaseed_local_denoise_report.py#L44) |
