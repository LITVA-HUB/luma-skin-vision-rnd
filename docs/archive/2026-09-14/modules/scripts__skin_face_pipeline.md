# `scripts/skin_face_pipeline.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_pipeline.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Finish this one ML workflow after the already-running training worker exits successfully.

SHA-256 исходника: `77761112b065c2fd03522ded5bb69451e16da29235cd23dd0e4418c5d7d451a8`. Строк: **76**.

## Зависимости

```python
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from skin_data_growth import DATA_ROOT, sha_bytes, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/skin_face_pipeline.py#L12)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 13](../../../../scripts/skin_face_pipeline.py#L13)

```python
OUT = DATA_ROOT/'facial_skin_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `wait_for_worker` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_face_pipeline.py#L16) |
| `main` | FunctionDef | См. реализацию | [L42](../../../../scripts/skin_face_pipeline.py#L42) |
