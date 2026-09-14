# `scripts/skin_capture_support_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_capture_support_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

TRAIN-only evidence for using actual same-site patch observations.

SHA-256 исходника: `3490e9c9246ccdd8b3d53a6e160a1c76e5abc9cbbaf455f74e385b90419173d0`. Строк: **55**.

## Зависимости

```python
import itertools,json
from pathlib import Path
import numpy as np
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_capture_support_audit.py#L9)

```python
OUT=ROOT/'docs/benchmarks/skin_capture_support_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_capture_support_audit.py#L12) |
