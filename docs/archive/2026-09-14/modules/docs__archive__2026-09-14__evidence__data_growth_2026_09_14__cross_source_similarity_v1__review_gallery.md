# `docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cross_source_similarity_v1/review_gallery.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cross_source_similarity_v1/review_gallery.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Display four highest-correlation candidate image pairs for content review.

SHA-256 исходника: `e9ba98d9ca6aca1340065ebb36c9e173d6fdaf377730e680ef71eebe5aaeb104`. Строк: **45**.

## Зависимости

```python
import json
from pathlib import Path
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cross_source_similarity_v1/review_gallery.py#L11)

```python
ROOT = Path(__file__).resolve().parent
```

[Строка 12](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cross_source_similarity_v1/review_gallery.py#L12)

```python
DATA = ROOT.parent
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `pixels` | FunctionDef | См. реализацию | [L19](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/cross_source_similarity_v1/review_gallery.py#L19) |
