# `docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Use author partitions and person codes; remove transitive duplicate leakage.

SHA-256 исходника: `f55c16efb34247fa5e88eb4803bbcc6a84dc8b53ee74149699c83a182d358f85`. Строк: **129**.

## Зависимости

```python
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py#L9)

```python
ROOT = Path(__file__).resolve().parent
```

[Строка 10](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py#L10)

```python
POOL = ROOT / 'prepared_192_v1'
```

[Строка 11](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py#L11)

```python
OUT = ROOT / 'split_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L14](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py#L14) |
| `write_json` | FunctionDef | См. реализацию | [L19](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py#L19) |
| `source_table` | FunctionDef | См. реализацию | [L25](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py#L25) |
| `main` | FunctionDef | См. реализацию | [L36](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/prepare_splits.py#L36) |
