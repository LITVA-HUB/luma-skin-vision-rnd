# `scripts/skin_face_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only Seg1 provenance and stored-metric arithmetic verification.

SHA-256 исходника: `59e0535ecd6a665b0c384318b9229cf1c11158111eefcc697b68caef08347c42`. Строк: **133**.

## Зависимости

```python
import hashlib
import json
import math
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_face_verify.py#L9)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 10](../../../../scripts/skin_face_verify.py#L10)

```python
DATA = Path('D:/Luma-RnD/data_growth_2026_09_14')
```

[Строка 11](../../../../scripts/skin_face_verify.py#L11)

```python
RUN = DATA / 'facial_skin_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_face_verify.py#L14) |
| `read` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_face_verify.py#L18) |
| `close` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_face_verify.py#L22) |
| `confusion` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_face_verify.py#L27) |
| `main` | FunctionDef | См. реализацию | [L37](../../../../scripts/skin_face_verify.py#L37) |
