# `scripts/cc_v7_external_population.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_external_population.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze metadata-only remainder population; no image decoding or GT parsing.

SHA-256 исходника: `e54b1d85cb4a98d4f2ea4cfbc0b9631f95fd722047974e96c9120d115695aecb`. Строк: **49**.

## Зависимости

```python
import json
from collections import Counter
from pathlib import Path
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/cc_v7_external_population.py#L9)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `select` | FunctionDef | См. реализацию | [L12](../../../../scripts/cc_v7_external_population.py#L12) |
| `run` | FunctionDef | См. реализацию | [L31](../../../../scripts/cc_v7_external_population.py#L31) |
