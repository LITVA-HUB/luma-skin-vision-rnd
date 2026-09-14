# `docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/inspect_originals.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/inspect_originals.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Validate original CAVE spectral planes without creating measured camera targets.

SHA-256 исходника: `9b35cc5808d31d645a0ddb9a1082a54a48875eb496cbff100119e3528a4686b3`. Строк: **74**.

## Зависимости

```python
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/inspect_originals.py#L12)

```python
ROOT = Path(__file__).resolve().parent
```

[Строка 13](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cave_skin_hair/inspect_originals.py#L13)

```python
OUT = ROOT / 'prepared_raw_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
