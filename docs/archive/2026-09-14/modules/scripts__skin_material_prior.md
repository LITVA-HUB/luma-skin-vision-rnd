# `scripts/skin_material_prior.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_material_prior.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze a measured-material prior and explicit observer/tail assumptions.

SHA-256 исходника: `9083b6a07c102ca525b2fa53541f741ca9b34b71c1170d98441d33e9d054a6d5`. Строк: **108**.

## Зависимости

```python
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scripts.skin_issa_material import load_cache,subject_weights,verify_lock
from scripts.skin_issa_data import sha,write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../scripts/skin_material_prior.py#L8)

```python
ROOT=Path(__file__).resolve().parents[1]
```

[Строка 14](../../../../scripts/skin_material_prior.py#L14)

```python
OUT=ROOT/'docs/benchmarks/skin_material_image_v1'
```

[Строка 15](../../../../scripts/skin_material_prior.py#L15)

```python
PRIOR=ROOT/'artifacts/skin_material_image_v1/prior.npz'
```

[Строка 16](../../../../scripts/skin_material_prior.py#L16)

```python
CIE=ROOT/'docs/data/provenance/cie_material_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `integration_matrix` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_material_prior.py#L19) |
| `lab_from_xyz` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_material_prior.py#L26) |
| `material_value_jacobian` | FunctionDef | См. реализацию | [L32](../../../../scripts/skin_material_prior.py#L32) |
| `original_cie` | FunctionDef | См. реализацию | [L39](../../../../scripts/skin_material_prior.py#L39) |
| `build` | FunctionDef | См. реализацию | [L57](../../../../scripts/skin_material_prior.py#L57) |
