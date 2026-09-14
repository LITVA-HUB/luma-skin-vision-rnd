# `scripts/cc_v7_external_lock.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_external_lock.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze all29 estimators/selectors before real V7 target image/GT decoding.

SHA-256 исходника: `4b1e9eae66b948bf498cfe18cf7377f0393010de232ea691e7404c46781ea6a4`. Строк: **120**.

## Зависимости

```python
import argparse
import copy
import json
from pathlib import Path
from cc_v7_verify import verify_manifest
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/cc_v7_external_lock.py#L12)

```python
ROOT=Path(__file__).resolve().parents[1]
```

[Строка 13](../../../../scripts/cc_v7_external_lock.py#L13)

```python
BENCH=ROOT/"docs/benchmarks/cc_v7_external"
```

[Строка 14](../../../../scripts/cc_v7_external_lock.py#L14)

```python
LOCK=BENCH/"method_lock.json"
```

[Строка 15](../../../../scripts/cc_v7_external_lock.py#L15)

```python
OLD_LOCK_SHA="f47778497a76782da288fe6740d8ef12388fa080ead9a08db618b7a6de393021"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `load` | FunctionDef | См. реализацию | [L18](../../../../scripts/cc_v7_external_lock.py#L18) |
| `freeze` | FunctionDef | См. реализацию | [L22](../../../../scripts/cc_v7_external_lock.py#L22) |
| `checked_lock` | FunctionDef | См. реализацию | [L101](../../../../scripts/cc_v7_external_lock.py#L101) |
