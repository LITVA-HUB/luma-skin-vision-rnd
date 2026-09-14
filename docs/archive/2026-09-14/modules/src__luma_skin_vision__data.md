# `src/luma_skin_vision/data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

One JSONL record per image and measured facial region; no identity features.

SHA-256 исходника: `963322016c73721917c4b9061a7375f21e125b9f6d4e0ca647b640088a139806`. Строк: **222**.

## Зависимости

```python
import hashlib
import json
from pathlib import Path
from typing import Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../src/luma_skin_vision/data.py#L11)

```python
SPLITS = ("train", "validation", "calibration", "test")
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Record` | BaseModel | [L14](../../../../src/luma_skin_vision/data.py#L14) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Record` | ClassDef | См. реализацию | [L14](../../../../src/luma_skin_vision/data.py#L14) |
| `sha256` | FunctionDef | См. реализацию | [L83](../../../../src/luma_skin_vision/data.py#L83) |
| `assign_splits` | FunctionDef | См. реализацию | [L91](../../../../src/luma_skin_vision/data.py#L91) |
| `resolve_image` | FunctionDef | См. реализацию | [L106](../../../../src/luma_skin_vision/data.py#L106) |
| `validate_records` | FunctionDef | См. реализацию | [L114](../../../../src/luma_skin_vision/data.py#L114) |
| `dataset_identity` | FunctionDef | См. реализацию | [L215](../../../../src/luma_skin_vision/data.py#L215) |
