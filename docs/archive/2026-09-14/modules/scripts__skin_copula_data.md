# `scripts/skin_copula_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_copula_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-only rank-dependence context, keeping absolute RGB in the main arm.

SHA-256 исходника: `666a0eb07547667e63d7f89cc34d3b6dbd876e1a43337f22be063f7ca235fa0a`. Строк: **74**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load as load_source_cache,features,CACHE as SOURCE_CACHE
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_copula_data.py#L9)

```python
CACHE=ROOT/'data/processed/skin_copula_v1'
```

[Строка 10](../../../../scripts/skin_copula_data.py#L10)

```python
OUT=ROOT/'docs/benchmarks/skin_copula_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `rank_channels` | FunctionDef | См. реализацию | [L13](../../../../scripts/skin_copula_data.py#L13) |
| `joint_hist` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_copula_data.py#L20) |
| `pack` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_copula_data.py#L27) |
| `load` | FunctionDef | См. реализацию | [L35](../../../../scripts/skin_copula_data.py#L35) |
| `main` | FunctionDef | См. реализацию | [L48](../../../../scripts/skin_copula_data.py#L48) |
