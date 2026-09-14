# `scripts/classical_public_cc.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/classical_public_cc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `2022226021c450db1d395797e2e7b7f0dd8c00afeb766aee88f47cfb8df8d588`. Строк: **27**.

## Зависимости

```python
import json
from pathlib import Path
from luma_skin_vision.cc.benchmark import indices, load
from luma_skin_vision.cc.core import EXPERT_NAMES, angular, reproduction, summarize
from luma_skin_vision.experiment import source_identity, write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
