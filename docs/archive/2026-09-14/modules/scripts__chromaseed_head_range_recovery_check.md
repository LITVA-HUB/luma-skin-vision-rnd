# `scripts/chromaseed_head_range_recovery_check.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_recovery_check.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Observe the replayed checkpoint and preserved banks without touching training.

SHA-256 исходника: `459e3b4174ded003339990ba386f8761b53f10f257f1d55ea9453403fb0f9f33`. Строк: **89**.

## Зависимости

```python
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/chromaseed_head_range_recovery_check.py#L9)

```python
RUN = Path('D:/Luma-RnD/chromaseed_head_range_v1')
```

[Строка 10](../../../../scripts/chromaseed_head_range_recovery_check.py#L10)

```python
OUT = RUN / 'recovery_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L13](../../../../scripts/chromaseed_head_range_recovery_check.py#L13) |
| `read` | FunctionDef | См. реализацию | [L18](../../../../scripts/chromaseed_head_range_recovery_check.py#L18) |
| `save` | FunctionDef | См. реализацию | [L22](../../../../scripts/chromaseed_head_range_recovery_check.py#L22) |
| `main` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_head_range_recovery_check.py#L28) |
