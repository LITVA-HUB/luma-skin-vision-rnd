# `scripts/cc_v2_figures.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v2_figures.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Publication-style figures from frozen per-image measurements, not new fitting.

SHA-256 исходника: `dae4d63866a57c2d312a15510e4b980308fb1b56dcde7cd073e003d70ba75374`. Строк: **137**.

## Зависимости

```python
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cc_v2_bootstrap import records
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/cc_v2_figures.py#L13)

```python
ROOT = Path("docs/benchmarks/cc_v2")
```

[Строка 14](../../../../scripts/cc_v2_figures.py#L14)

```python
FIGURES = ROOT / "figures"
```

[Строка 16](../../../../scripts/cc_v2_figures.py#L16)

```python
METHODS = [
    ("ccv2_sog_large_g0::combined", "Proposed: SoG residual, 3.03M", "#0072B2"),
    ("ccv2_direct_large_g0::combined", "Matched C+ with combined head, 3.03M", "#D55E00"),
    ("gw_ridge1::cheap", "GW + ridge, statistics baseline", "#009E73"),
    ("ccv2_legacy_proposed::upgraded_combined", "v1 mixture + upgraded error head", "#CC79A7"),
    ("ccv2_legacy_baseline::v1_shades_gray", "Classical SoG + learned selector", "#666666"),
]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
