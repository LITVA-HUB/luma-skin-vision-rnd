# `scripts/skin_uminho_train_expand_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_uminho_train_expand_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Acquire all original TRAIN reflectance cubes; preserve the old held allocation.

SHA-256 исходника: `5bfc8e6374c0a828e2f93909a6e89f6d06d3c7bebeae7fb68ba48a7459ee90c5`. Строк: **124**.

## Зависимости

```python
import hashlib
import json
import time
from pathlib import Path
from urllib.request import urlopen
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../scripts/skin_uminho_train_expand_v2.py#L8)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 9](../../../../scripts/skin_uminho_train_expand_v2.py#L9)

```python
LEGACY = Path('C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/public/uminho_hsfd_v1')
```

[Строка 10](../../../../scripts/skin_uminho_train_expand_v2.py#L10)

```python
OUT = Path('D:/Luma-RnD/data_growth_2026_09_14/uminho_train_v2')
```

[Строка 11](../../../../scripts/skin_uminho_train_expand_v2.py#L11)

```python
MANIFEST_SHA = '3d3b7dc7256eb04250dee2f5ef76880f4101d019313d97ba051480f8d320b5c7'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `training_rows` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_uminho_train_expand_v2.py#L14) |
| `digest` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_uminho_train_expand_v2.py#L27) |
| `save_once` | FunctionDef | См. реализацию | [L35](../../../../scripts/skin_uminho_train_expand_v2.py#L35) |
| `checked_file` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_uminho_train_expand_v2.py#L44) |
| `main` | FunctionDef | См. реализацию | [L64](../../../../scripts/skin_uminho_train_expand_v2.py#L64) |
