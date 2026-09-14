# `src/luma_skin_vision/cc/core.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/cc/core.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `b6bbb3237349913de247bdf5f788cfc0ac415936af42df48a67fdb869264fcac`. Строк: **106**.

## Зависимости

```python
import hashlib
import cv2
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../src/luma_skin_vision/cc/core.py#L6)

```python
EXPERT_NAMES = ["gray_world", "max_rgb", "shades_gray", "gray_edge"]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `unit` | FunctionDef | См. реализацию | [L9](../../../../src/luma_skin_vision/cc/core.py#L9) |
| `angular` | FunctionDef | См. реализацию | [L17](../../../../src/luma_skin_vision/cc/core.py#L17) |
| `reproduction` | FunctionDef | См. реализацию | [L21](../../../../src/luma_skin_vision/cc/core.py#L21) |
| `summarize` | FunctionDef | См. реализацию | [L29](../../../../src/luma_skin_vision/cc/core.py#L29) |
| `selective_curve` | FunctionDef | См. реализацию | [L50](../../../../src/luma_skin_vision/cc/core.py#L50) |
| `linearize` | FunctionDef | См. реализацию | [L72](../../../../src/luma_skin_vision/cc/core.py#L72) |
| `experts` | FunctionDef | См. реализацию | [L83](../../../../src/luma_skin_vision/cc/core.py#L83) |
