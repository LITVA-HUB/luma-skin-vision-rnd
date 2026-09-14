# `scripts/chromaseed_architecture_scale_support.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_architecture_scale_support.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Inner-only diagnostic of the fixed +/-one-target-std residual output range.

SHA-256 исходника: `0e15bece7384af7b109d895d27dffa2c94f8b1ae71c04d4321bbc5b8b2fe2240`. Строк: **152**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, TIMES, VARIANTS, bank_path, load_data
from chromaseed_gated_audit import model_from
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_prefix_numpy import predict as warm_predict
from chromaseed_widen_run import bank_path as warm_bank
from skin_local_search_train import roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/chromaseed_architecture_scale_support.py#L13)

```python
MARGIN = 1e-4
```

[Строка 14](../../../../scripts/chromaseed_architecture_scale_support.py#L14)

```python
NEAR = 0.95
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `person_average` | FunctionDef | Equal-person means, then equal means over the three separate seeds. | [L17](../../../../scripts/chromaseed_architecture_scale_support.py#L17) |
| `distance_to_box` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_architecture_scale_support.py#L22) |
| `main` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_architecture_scale_support.py#L28) |
