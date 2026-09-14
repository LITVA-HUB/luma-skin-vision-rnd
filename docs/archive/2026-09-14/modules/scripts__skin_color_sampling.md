# `scripts/skin_color_sampling.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_color_sampling.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Selected-TRAIN native-color allocation; no evaluation labels or camera input.

SHA-256 исходника: `54ca5baa8724f430de6af56b5eef0b28a9363cd8b00402c34b962ebc41e3cd3c`. Строк: **35**.

## Зависимости

```python
import numpy as np
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_color_sampling.py#L5)

```python
ARMS=('image','person_site','site','color','color_ipw')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sampling_distribution` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_color_sampling.py#L8) |
| `draw_indices` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_color_sampling.py#L31) |
