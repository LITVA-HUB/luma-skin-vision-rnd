# `src/luma_skin_vision/export/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/export/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

FP32 export verification only; production optimization awaits real evidence.

SHA-256 исходника: `e5528e0d949be7e713bad3a52e56a6840accab92570fd8be6f67b3db41ce8c7c`. Строк: **191**.

## Зависимости

```python
import time
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ort_inputs` | FunctionDef | См. реализацию | [L12](../../../../src/luma_skin_vision/export/__init__.py#L12) |
| `export_model` | FunctionDef | См. реализацию | [L20](../../../../src/luma_skin_vision/export/__init__.py#L20) |
| `export_run` | FunctionDef | См. реализацию | [L62](../../../../src/luma_skin_vision/export/__init__.py#L62) |
| `benchmark_run` | FunctionDef | См. реализацию | [L93](../../../../src/luma_skin_vision/export/__init__.py#L93) |
