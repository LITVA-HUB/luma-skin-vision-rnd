# `scripts/skin_local_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Standard local-reference regressions; no query references or camera identity.

SHA-256 исходника: `8c553a9fba42dea6dd6f36ca786e19656bb60aac62f4b3d67e038c1ad32559a9`. Строк: **40**.

## Зависимости

```python
import numpy as np
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_local_reference.py#L5)

```python
METHODS=('global_ridge','global_mean','appearance_mean','appearance_affine','color_mean','color_affine')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `normalized_weights` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_local_reference.py#L8) |
| `local_affine` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_local_reference.py#L14) |
| `predict_bank` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_local_reference.py#L22) |
