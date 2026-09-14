# `scripts/skin_capture_support.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_capture_support.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Actual observed patch-bag mixing within a fixed instrument-reference site.

SHA-256 исходника: `44423119c5b289cfea026555f9a24957df62e4075b22495611b6969a07580499`. Строк: **34**.

## Зависимости

```python
import numpy as np
import torch
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_capture_support.py#L5)

```python
ARMS=['baseline','self_bootstrap','soft_mode_control','paired_union','paired_stratified']
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `make_plan` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_capture_support.py#L8) |
| `apply_plan` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_capture_support.py#L15) |
