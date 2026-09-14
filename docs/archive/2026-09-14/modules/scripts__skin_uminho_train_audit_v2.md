# `scripts/skin_uminho_train_audit_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_uminho_train_audit_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Numerical inventory of the 19 acquired TRAIN cubes, without RGB rendering.

SHA-256 исходника: `5f9bc4005b6180c54c61d355e7ddfc9aaf7ba3752be8f9db5240c09c3820a2d3`. Строк: **53**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skin_uminho_train_expand_v2 import OUT, digest, save_once
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_uminho_train_audit_v2.py#L10) |
