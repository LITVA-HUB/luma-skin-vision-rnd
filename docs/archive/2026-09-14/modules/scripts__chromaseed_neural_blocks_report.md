# `scripts/chromaseed_neural_blocks_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_blocks_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal NB negative equivalence evidence; later calls only verify.

SHA-256 исходника: `f5c8dd365d5dc1ab16c852677a6cdb1929ea9346c1a6cb81ad25667f407577b2`. Строк: **292**.

## Зависимости

```python
from __future__ import annotations
import csv
import json
import subprocess
import sys
from chromaseed_kernel_audit import js
from chromaseed_neural_blocks_run import OUT, ROOT, RUN, check_map
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_neural_blocks_report.py#L14)

```python
CARD = ROOT / "docs/architecture/chromaseed_neural_blocks_model_card.md"
```

[Строка 15](../../../../scripts/chromaseed_neural_blocks_report.py#L15)

```python
NEXT = ROOT / "docs/research/chromaseed_neural_blocks_next_decision.md"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `csv_out` | FunctionDef | См. реализацию | [L18](../../../../scripts/chromaseed_neural_blocks_report.py#L18) |
| `main` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_neural_blocks_report.py#L31) |
