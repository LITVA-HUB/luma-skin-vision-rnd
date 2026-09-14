# `scripts/skin_mskcc_selective_core.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_selective_core.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Native-color inference, interpretable risk features and immutable bindings.

SHA-256 исходника: `a5151967444791123ee5a3a70707419313cc0e187f06a091b2fd8538ecc052b4`. Строк: **87**.

## Зависимости

```python
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from skin_mskcc_data import ROOT,MANIFEST,sha
from skin_mskcc_vote import PatchVotes
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/skin_mskcc_selective_core.py#L12)

```python
PROTOCOL=ROOT/'docs/research/skin_mskcc_selective_protocol_v1.md'
```

[Строка 13](../../../../scripts/skin_mskcc_selective_core.py#L13)

```python
OUT=ROOT/'docs/benchmarks/skin_mskcc_selective_v1'
```

[Строка 14](../../../../scripts/skin_mskcc_selective_core.py#L14)

```python
RUN=ROOT/'experiments/runs/skin_mskcc_selective_v1'
```

[Строка 15](../../../../scripts/skin_mskcc_selective_core.py#L15)

```python
SEEDS=[17,29,43]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `patient_folds` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_mskcc_selective_core.py#L18) |
| `density` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_mskcc_selective_core.py#L24) |
| `plain_predict` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_mskcc_selective_core.py#L29) |
| `designs` | FunctionDef | См. реализацию | [L52](../../../../scripts/skin_mskcc_selective_core.py#L52) |
| `deployment_designs` | FunctionDef | См. реализацию | [L66](../../../../scripts/skin_mskcc_selective_core.py#L66) |
| `binding` | FunctionDef | См. реализацию | [L74](../../../../scripts/skin_mskcc_selective_core.py#L74) |
| `verify_lock` | FunctionDef | См. реализацию | [L78](../../../../scripts/skin_mskcc_selective_core.py#L78) |
