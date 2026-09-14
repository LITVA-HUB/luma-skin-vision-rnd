# `scripts/skin_gradient_transfer.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_gradient_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

TRAIN-only group-gradient falsifier, not a new accuracy experiment.

SHA-256 исходника: `8460b478ff989797173a11d6139927801c39bc767d18e4edf364c61bea78e1b9`. Строк: **54**.

## Зависимости

```python
from contextlib import contextmanager
import numpy as np
import torch
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../scripts/skin_gradient_transfer.py#L6)

```python
SHUFFLES=(51871,51872,51873)
```

[Строка 7](../../../../scripts/skin_gradient_transfer.py#L7)

```python
STEPS=(1e-4,1e-3)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `partitions` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_gradient_transfer.py#L10) |
| `group_means` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_gradient_transfer.py#L15) |
| `gradient_statistics` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_gradient_transfer.py#L19) |
| `transient_step` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_gradient_transfer.py#L29) |
| `component_gradients` | FunctionDef | См. реализацию | [L43](../../../../scripts/skin_gradient_transfer.py#L43) |
