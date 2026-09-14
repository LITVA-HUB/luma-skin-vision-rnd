# `scripts/archive_cc_v2_metadata.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/archive_cc_v2_metadata.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Preserve exact training/selection metadata and code bytes without copying datasets/weights.

SHA-256 исходника: `bc9a8ec51faec33bdf5204567593ab8df09138b9581876f986f17b6a76f0715d`. Строк: **87**.

## Зависимости

```python
import importlib.metadata
import json
from pathlib import Path
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L11](../../../../scripts/archive_cc_v2_metadata.py#L11) |
