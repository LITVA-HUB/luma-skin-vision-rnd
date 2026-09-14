# `scripts/report_public_cc.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/report_public_cc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Build source-backed report; no training or selection choices occur here.

SHA-256 исходника: `32093048ea27cc3a19ba868e90ebde40aa8d0896c9869f25fdcf9f61302a532a`. Строк: **361**.

## Зависимости

```python
import hashlib
import json
import re
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from luma_skin_vision.cc.benchmark import indices, load
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/report_public_cc.py#L18)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 19](../../../../scripts/report_public_cc.py#L19)

```python
OUT = ROOT / "docs/benchmarks"
```

[Строка 20](../../../../scripts/report_public_cc.py#L20)

```python
SEEDS = [17, 29, 43]
```

[Строка 21](../../../../scripts/report_public_cc.py#L21)

```python
LABELS = {
    "gray_world": "Gray World + learned selector",
    "max_rgb": "Max RGB + learned selector",
    "shades_gray": "Shades of Gray + learned selector",
    "gray_edge": "Gray Edge + learned selector",
    "baseline_context": "C+ standard context error head",
    "baseline_disagreement": "C + disagreement-only head",
    "baseline_combined": "C + combined error head (strong control)",
    "proposed_context": "Mixture + context-only head",
    "proposed_disagreement": "Mixture + disagreement-only head",
    "proposed_combined": "Proposed mixture + combined head",
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `run` | FunctionDef | См. реализацию | [L35](../../../../scripts/report_public_cc.py#L35) |
| `values` | FunctionDef | См. реализацию | [L43](../../../../scripts/report_public_cc.py#L43) |
| `avg_sd` | FunctionDef | См. реализацию | [L48](../../../../scripts/report_public_cc.py#L48) |
| `boot_pair` | FunctionDef | См. реализацию | [L52](../../../../scripts/report_public_cc.py#L52) |
| `main` | FunctionDef | См. реализацию | [L78](../../../../scripts/report_public_cc.py#L78) |
