# `apps/training-dashboard/palette_transfer_monitor.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../apps/training-dashboard/palette_transfer_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only P3 telemetry; no training imports, GPU work or inferred completion.

SHA-256 исходника: `184ad523794b462b8c97387c2c84243d37c0ba3eb94d7ed434112473432e2c37`. Строк: **321**.

## Зависимости

```python
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from monitor import LABELS, ROLES, process_alive
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../apps/training-dashboard/palette_transfer_monitor.py#L12)

```python
VARIANTS = ("patch5m", "soft5m", "dynamic5m")
```

[Строка 13](../../../../apps/training-dashboard/palette_transfer_monitor.py#L13)

```python
ARMS = ("aligned", "shuffled")
```

[Строка 14](../../../../apps/training-dashboard/palette_transfer_monitor.py#L14)

```python
PROJECT = Path(__file__).resolve().parents[2]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `positive` | FunctionDef | См. реализацию | [L17](../../../../apps/training-dashboard/palette_transfer_monitor.py#L17) |
| `snapshot` | FunctionDef | См. реализацию | [L26](../../../../apps/training-dashboard/palette_transfer_monitor.py#L26) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>gate · L71–77</summary>

```python
def gate(value, device):
        return bool(
            registration_sha
            and value.get("passed") is True
            and value.get("registration_sha256") == registration_sha
            and value.get("device") == device
        )
```

</details>
