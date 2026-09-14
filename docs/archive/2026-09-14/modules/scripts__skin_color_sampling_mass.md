# `scripts/skin_color_sampling_mass.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_color_sampling_mass.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Preserve person probability mass while removing within-person color emphasis.

SHA-256 исходника: `3527a04744f93fb435fbbe93863a3f7bdd6a1c1f364ece32f9ebf0aa61059332`. Строк: **17**.

## Зависимости

```python
import numpy as np
from skin_color_sampling import sampling_distribution as base_distribution,draw_indices
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_color_sampling_mass.py#L5)

```python
ARMS=('person_mass','within_person_shuffle')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sampling_distribution` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_color_sampling_mass.py#L8) |
