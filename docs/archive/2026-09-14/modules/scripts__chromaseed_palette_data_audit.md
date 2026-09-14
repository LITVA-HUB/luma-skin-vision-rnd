# `scripts/chromaseed_palette_data_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_data_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Original spatial-source and photometric-view readback before auxiliary fitting.

SHA-256 исходника: `313e6a0a7a81ac330b7ee8c8427360b8e2d9b367bfb26f050ed8544f97835767`. Строк: **121**.

## Зависимости

```python
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from chromaseed_palette_data import OUT, P1, ROOT, read, save, sha, verify
from PIL import Image
from scipy.io import loadmat
from skin_mskcc_pixels import features
from skin_spectral_palette_audit import reference_integration
from threadpoolctl import threadpool_limits
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `statistics` | FunctionDef | См. реализацию | [L16](../../../../scripts/chromaseed_palette_data_audit.py#L16) |
| `main` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_palette_data_audit.py#L30) |
