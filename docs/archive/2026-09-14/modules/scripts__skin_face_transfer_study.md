# `scripts/skin_face_transfer_study.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_transfer_study.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed Seg2 comparison, validation-only choices and immutable experiment bindings.

SHA-256 исходника: `231b13145b311288bedd12403e9de43fecc6204fc03fa5e322ea6da03d83c406`. Строк: **218**.

## Зависимости

```python
from __future__ import annotations
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from skin_face_transfer_data import DATA_ROOT, SOURCES, checked, digest, read
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/skin_face_transfer_study.py#L12)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 13](../../../../scripts/skin_face_transfer_study.py#L13)

```python
RUN = Path('D:/Luma-RnD/skin_face_transfer_v1')
```

[Строка 14](../../../../scripts/skin_face_transfer_study.py#L14)

```python
SEG1 = DATA_ROOT / 'facial_skin_v1'
```

[Строка 15](../../../../scripts/skin_face_transfer_study.py#L15)

```python
INITIAL = SEG1 / 'best.pt'
```

[Строка 16](../../../../scripts/skin_face_transfer_study.py#L16)

```python
INITIAL_SHA = '1203cbc5ed2ee17cb2a408c47a23b3a28468f169d48b4d3cee8bd3059e7fbed3'
```

[Строка 17](../../../../scripts/skin_face_transfer_study.py#L17)

```python
ARMS = ('lapa_only', 'lapa_celeba')
```

[Строка 18](../../../../scripts/skin_face_transfer_study.py#L18)

```python
SEEDS = (17, 29, 43)
```

[Строка 19](../../../../scripts/skin_face_transfer_study.py#L19)

```python
BUDGETS = (1494, 2988, 5976)
```

[Строка 21](../../../../scripts/skin_face_transfer_study.py#L21)

```python
COUNTS = {'lapa': {'train': 15914, 'validation': 1692, 'test': 2000},
          'celeba': {'train': 24112, 'validation': 2992, 'test': 2822}}
```

[Строка 23](../../../../scripts/skin_face_transfer_study.py#L23)

```python
FILES = (
    'scripts/skin_face_transfer_data.py',
    'scripts/skin_face_transfer_study.py',
    'scripts/skin_face_transfer_run.py',
    'scripts/skin_face_transfer_evaluate.py',
    'scripts/skin_face_transfer_report.py',
    'tests/test_skin_face_transfer_data.py',
    'tests/test_skin_face_transfer_study.py',
    'tests/test_skin_face_transfer_run.py',
    'tests/test_skin_face_transfer_evaluate.py',
    'docs/superpowers/specs/2026-09-14-face-transfer-design.md',
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `utc` | FunctionDef | См. реализацию | [L37](../../../../scripts/skin_face_transfer_study.py#L37) |
| `write_once` | FunctionDef | См. реализацию | [L41](../../../../scripts/skin_face_transfer_study.py#L41) |
| `check_bindings` | FunctionDef | См. реализацию | [L49](../../../../scripts/skin_face_transfer_study.py#L49) |
| `recipe` | FunctionDef | См. реализацию | [L55](../../../../scripts/skin_face_transfer_study.py#L55) |
| `validation_score` | FunctionDef | См. реализацию | [L79](../../../../scripts/skin_face_transfer_study.py#L79) |
| `select_prefix` | FunctionDef | См. реализацию | [L96](../../../../scripts/skin_face_transfer_study.py#L96) |
| `freeze_choices` | FunctionDef | См. реализацию | [L119](../../../../scripts/skin_face_transfer_study.py#L119) |
| `initial_bindings` | FunctionDef | См. реализацию | [L142](../../../../scripts/skin_face_transfer_study.py#L142) |
| `data_bindings` | FunctionDef | См. реализацию | [L161](../../../../scripts/skin_face_transfer_study.py#L161) |
| `current_bindings` | FunctionDef | См. реализацию | [L186](../../../../scripts/skin_face_transfer_study.py#L186) |
| `freeze_registration` | FunctionDef | См. реализацию | [L203](../../../../scripts/skin_face_transfer_study.py#L203) |
| `verify_registration` | FunctionDef | См. реализацию | [L213](../../../../scripts/skin_face_transfer_study.py#L213) |
