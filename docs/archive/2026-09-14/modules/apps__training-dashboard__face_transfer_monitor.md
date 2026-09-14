# `apps/training-dashboard/face_transfer_monitor.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../apps/training-dashboard/face_transfer_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only Seg2 monitor: JSON metadata and process state, never ML/data-array imports.

SHA-256 исходника: `c7e8f345847b03fdec8c3bdf0b4df4b7827475b0ca34fed803928b7412755a88`. Строк: **282**.

## Зависимости

```python
import hashlib
import json
import math
import time
from pathlib import Path
from monitor import process_alive
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../apps/training-dashboard/face_transfer_monitor.py#L10)

```python
PROJECT = Path(__file__).resolve().parents[2]
```

[Строка 11](../../../../apps/training-dashboard/face_transfer_monitor.py#L11)

```python
ARMS = ('lapa_only','lapa_celeba')
```

[Строка 12](../../../../apps/training-dashboard/face_transfer_monitor.py#L12)

```python
SEEDS = (17,29,43)
```

[Строка 14](../../../../apps/training-dashboard/face_transfer_monitor.py#L14)

```python
COUNTS = {'lapa':{'train':15914,'validation':1692,'test':2000},
          'celeba':{'train':24112,'validation':2992,'test':2822}}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `finite` | FunctionDef | См. реализацию | [L18](../../../../apps/training-dashboard/face_transfer_monitor.py#L18) |
| `integer` | FunctionDef | См. реализацию | [L22](../../../../apps/training-dashboard/face_transfer_monitor.py#L22) |
| `snapshot` | FunctionDef | См. реализацию | [L26](../../../../apps/training-dashboard/face_transfer_monitor.py#L26) |
