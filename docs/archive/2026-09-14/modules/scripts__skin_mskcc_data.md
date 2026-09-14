# `scripts/skin_mskcc_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Original MSKCC joins and patient roles. No endpoint values in manifest.

SHA-256 исходника: `12f7ce2df93ca3586fa068d92751670c33372f7f69bad4d2ca550463e4de0fc5`. Строк: **116**.

## Зависимости

```python
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../scripts/skin_mskcc_data.py#L8)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 9](../../../../scripts/skin_mskcc_data.py#L9)

```python
RAW = ROOT / "data/public/mskcc_skin_v1"
```

[Строка 10](../../../../scripts/skin_mskcc_data.py#L10)

```python
PROV = ROOT / "docs/data/provenance/mskcc_skin_v1"
```

[Строка 11](../../../../scripts/skin_mskcc_data.py#L11)

```python
PROTOCOL = ROOT / "docs/research/skin_mskcc_protocol_v1.md"
```

[Строка 12](../../../../scripts/skin_mskcc_data.py#L12)

```python
MANIFEST = ROOT / "data/processed/skin_mskcc_v1/manifest.json"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_mskcc_data.py#L15) |
| `read` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_mskcc_data.py#L19) |
| `patient_roles` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_mskcc_data.py#L24) |
| `prepare` | FunctionDef | См. реализацию | [L39](../../../../scripts/skin_mskcc_data.py#L39) |
| `manifest` | FunctionDef | См. реализацию | [L82](../../../../scripts/skin_mskcc_data.py#L82) |
| `load_source` | FunctionDef | См. реализацию | [L92](../../../../scripts/skin_mskcc_data.py#L92) |
