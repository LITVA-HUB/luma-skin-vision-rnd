# `scripts/cc_v6_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v6_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Archive all completed V6 seeds and summarize the negative combination screen.

SHA-256 исходника: `0783cdb0fa2c27d9f966229f0d52ab42e0af71aca3eb712e511dc4b5e9d8d2ec`. Строк: **62**.

## Зависимости

```python
import json
import shutil
from pathlib import Path
import numpy as np
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/cc_v6_report.py#L11)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `run` | FunctionDef | См. реализацию | [L14](../../../../scripts/cc_v6_report.py#L14) |
