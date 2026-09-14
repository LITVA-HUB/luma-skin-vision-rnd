# `scripts/skin_mskcc_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar CIEDE2000 + replay audit; source validation only.

SHA-256 исходника: `bd646188e042b1f5727f7f96b769b02de83a45951242f1d2d269e1b691d9846a`. Строк: **94**.

## Зависимости

```python
import json
import math as m
from pathlib import Path
import joblib
import numpy as np
from skin_mskcc_data import ROOT, MANIFEST, PROTOCOL, load_source, sha
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `scalar_de` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_mskcc_audit.py#L10) |
| `main` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_mskcc_audit.py#L44) |
