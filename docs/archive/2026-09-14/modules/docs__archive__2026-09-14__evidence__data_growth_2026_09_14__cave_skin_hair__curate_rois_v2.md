# `docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/curate_rois_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/curate_rois_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Small visually inspected material ROIs; keep the printed face out of live-skin data.

SHA-256 исходника: `b6258179bd8b75cda8906e2561d969bd614466a5ef8bd7474dc70f5cfcbd058c`. Строк: **79**.

## Зависимости

```python
import hashlib
import json
from collections import Counter
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/curate_rois_v2.py#L13)

```python
ROOT = Path(__file__).resolve().parent / 'prepared_raw_v1'
```

[Строка 14](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/curate_rois_v2.py#L14)

```python
OUT = ROOT / 'material_rois_v2'
```

[Строка 16](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/curate_rois_v2.py#L16)

```python
ROIS = [
    ('face', 'visible_skin', (280, 165, 352, 197)),
    ('face', 'visible_skin', (225, 286, 266, 328)),
    ('face', 'visible_skin', (367, 285, 401, 324)),
    ('photo_and_face', 'visible_skin', (383, 240, 420, 266)),
    ('photo_and_face', 'visible_skin', (347, 314, 371, 338)),
    ('photo_and_face', 'visible_skin', (433, 311, 455, 335)),
    ('photo_and_face', 'printed_face', (85, 280, 123, 306)),
    ('photo_and_face', 'printed_face', (54, 339, 76, 365)),
    ('photo_and_face', 'printed_face', (120, 339, 136, 363)),
    ('hairs', 'hair_appearance', (65, 180, 145, 340)),
    ('hairs', 'hair_appearance', (224, 175, 288, 335)),
    ('hairs', 'hair_appearance', (360, 165, 408, 325)),
]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
