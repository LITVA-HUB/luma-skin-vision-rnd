# `scripts/skin_sampling_transfer.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_sampling_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Combine person coverage and within-person native-color allocation.

SHA-256 исходника: `baa6b71513da9b6a3d0d120338c8ec91eb37930bf8ac153bdb98c89010084d21`. Строк: **24**.

## Зависимости

```python
import numpy as np
from skin_color_sampling import sampling_distribution as base_distribution,draw_indices
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_sampling_transfer.py#L5)

```python
ARMS=('image','person_site','site','color','color_ipw','person_color')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sampling_distribution` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_sampling_transfer.py#L8) |
| `protocols` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_sampling_transfer.py#L17) |
