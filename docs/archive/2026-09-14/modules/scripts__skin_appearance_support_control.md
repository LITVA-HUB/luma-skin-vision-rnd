# `scripts/skin_appearance_support_control.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_appearance_support_control.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen direct-color support restriction, with independent scalar audit.

SHA-256 исходника: `467eaf47a81491c2c1bf9c4c2fc28a495bfce302b02a615567fdeeb6789b576d`. Строк: **78**.

## Зависимости

```python
import argparse,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_appearance_inverse_train import OUT,RUN,ALPHAS
from skin_color_hull_control import make_hull,project_hull
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write,rows
from skin_distribution_train import score
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/skin_appearance_support_control.py#L16)

```python
PROTOCOL=ROOT/'docs/research/skin_appearance_support_control_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_appearance_support_control.py#L19) |
