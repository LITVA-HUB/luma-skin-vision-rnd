# `scripts/run_smoke.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/run_smoke.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Run actual CLI entry points; preserve commands and outputs under artifacts.

SHA-256 исходника: `69dae2d50d81217329515a6e02d49d72c8a06a5c24efdfb443d477d7a4c22e47`. Строк: **91**.

## Зависимости

```python
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/run_smoke.py#L10)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L13](../../../../scripts/run_smoke.py#L13) |
