# `src/luma_skin_vision/cli.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/cli.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `7dc13a5820095d02b9683054fa1f231f0f991835922cce7acaefcfccf7b995d2`. Строк: **132**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import yaml
from luma_skin_vision.data import Record, assign_splits, validate_records
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L11](../../../../src/luma_skin_vision/cli.py#L11) |
