# `scripts/build_archive_figures.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/build_archive_figures.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Render source-backed archive figures and explicit AS layer cards; no model execution.

SHA-256 исходника: `45b1249662e796fdab5ad3b8031561d071fda15b79b6f5334a96a329b679045a`. Строк: **287**.

## Зависимости

```python
import hashlib
import json
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/build_archive_figures.py#L13)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 14](../../../../scripts/build_archive_figures.py#L14)

```python
OUT = ROOT / "docs/archive/2026-09-14"
```

[Строка 15](../../../../scripts/build_archive_figures.py#L15)

```python
FIG = OUT / "figures"
```

[Строка 16](../../../../scripts/build_archive_figures.py#L16)

```python
SOURCES = {}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `load` | FunctionDef | См. реализацию | [L19](../../../../scripts/build_archive_figures.py#L19) |
| `save` | FunctionDef | См. реализацию | [L25](../../../../scripts/build_archive_figures.py#L25) |
| `specs` | FunctionDef | См. реализацию | [L42](../../../../scripts/build_archive_figures.py#L42) |
| `main` | FunctionDef | См. реализацию | [L60](../../../../scripts/build_archive_figures.py#L60) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>specs · L42–57</summary>

```python
def specs(variant):
    if variant == "pool5m":
        return [("token1", 18, 512), ("mlp1", 1572, 3072), ("head", 3072, 3)]
    a, e, h = (32, 24, 48) if variant.endswith("small") else (384, 256, 1120)
    layers = [("token1", 18, a), ("token2", a, e)]
    if variant.startswith("patch"):
        h1, h2, h3 = (96, 64, 48) if variant.endswith("small") else (1792, 1536, 1024)
        return layers + [("mlp1", 36 + e, h1), ("mlp2", h1, h2), ("mlp3", h2, h3), ("head", h3, 3)]
    return layers + [
        ("context", 36 + e, h),
        ("query", h + 3, e),
        ("key", e, e),
        ("update1", 2 * h + e + 3, h),
        ("update2", h, h),
        ("head", h, 3),
    ]
```

</details>
