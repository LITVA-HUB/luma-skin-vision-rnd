# `scripts/skin_issa_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_issa_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only original ISSA extraction; numerical endpoint access is role gated.

SHA-256 исходника: `bd7f796c312c3a46553ba0efcf8090ea12c6db81b3432b9864b76dc6ccc0fc5f`. Строк: **178**.

## Зависимости

```python
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/skin_issa_data.py#L10)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 11](../../../../scripts/skin_issa_data.py#L11)

```python
RAW = ROOT / 'data/public/issa_v4/ISSA_17_Jan_2025_Yan_Lu.xlsx'
```

[Строка 12](../../../../scripts/skin_issa_data.py#L12)

```python
RAW_SHA = '7396aa7608f1a6f91ec70bf8f8421252a9d81f71a2bca066eb7e98155b9d1161'
```

[Строка 13](../../../../scripts/skin_issa_data.py#L13)

```python
PRIVATE = ROOT / 'data/processed/skin_issa_v1'
```

[Строка 14](../../../../scripts/skin_issa_data.py#L14)

```python
OUT = ROOT / 'docs/benchmarks/skin_issa_v1'
```

[Строка 15](../../../../scripts/skin_issa_data.py#L15)

```python
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_issa_data.py#L18) |
| `write_json` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_issa_data.py#L22) |
| `column_name` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_issa_data.py#L27) |
| `selected_cells` | FunctionDef | См. реализацию | [L37](../../../../scripts/skin_issa_data.py#L37) |
| `original_xml` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_issa_data.py#L50) |
| `metadata` | FunctionDef | См. реализацию | [L59](../../../../scripts/skin_issa_data.py#L59) |
| `assign_roles` | FunctionDef | См. реализацию | [L77](../../../../scripts/skin_issa_data.py#L77) |
| `checked_manifest` | FunctionDef | См. реализацию | [L105](../../../../scripts/skin_issa_data.py#L105) |
| `endpoint_rows` | FunctionDef | См. реализацию | [L113](../../../../scripts/skin_issa_data.py#L113) |
| `constants` | FunctionDef | См. реализацию | [L127](../../../../scripts/skin_issa_data.py#L127) |
| `main` | FunctionDef | См. реализацию | [L134](../../../../scripts/skin_issa_data.py#L134) |
