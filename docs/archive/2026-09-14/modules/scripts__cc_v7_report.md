# `scripts/cc_v7_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v7_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Archive a completed V7 seed with standard source-risk and virtual-sensor results.

SHA-256 исходника: `ce419ecaa6cac90e160b062f03232feac25a29b1a7d50b9fbf673df92cbf825a`. Строк: **67**.

## Зависимости

```python
import argparse
import json
import shutil
from pathlib import Path
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/cc_v7_report.py#L10)

```python
ROOT=Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `archive` | FunctionDef | См. реализацию | [L13](../../../../scripts/cc_v7_report.py#L13) |
| `run` | FunctionDef | См. реализацию | [L27](../../../../scripts/cc_v7_report.py#L27) |
