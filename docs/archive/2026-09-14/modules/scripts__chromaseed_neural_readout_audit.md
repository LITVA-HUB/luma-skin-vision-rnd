# `scripts/chromaseed_neural_readout_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_readout_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent NR representation retraining, all-head QR reconstruction and output audit.

SHA-256 исходника: `4c71022ec65d100ec142dcd48b6d5d03a0bb368869004408624c8d1644a26478`. Строк: **568**.

## Зависимости

```python
from __future__ import annotations
import argparse
import time
from collections import Counter
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact, summaries
from chromaseed_feature_groups_audit import compare
from chromaseed_gate_stability_audit import close, transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer, direct, ident, row_hash, scoring
from chromaseed_gaussian_reference import initial, normalization, train_reference
from chromaseed_kernel_audit import js, nz
from chromaseed_neural_readout_reference import refit
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/chromaseed_neural_readout_audit.py#L23)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 24](../../../../scripts/chromaseed_neural_readout_audit.py#L24)

```python
TG = ROOT / "experiments/runs/chromaseed_gaussian_v1"
```

[Строка 25](../../../../scripts/chromaseed_neural_readout_audit.py#L25)

```python
SOURCE = "70246d9935e52d5be0dcebf758ba3e63b30be4965a9154a19ba180f5cf1782c1"
```

[Строка 26](../../../../scripts/chromaseed_neural_readout_audit.py#L26)

```python
SEEDS = (17, 29, 43)
```

[Строка 27](../../../../scripts/chromaseed_neural_readout_audit.py#L27)

```python
EPOCHS = (1, 4, 16)
```

[Строка 28](../../../../scripts/chromaseed_neural_readout_audit.py#L28)

```python
GROUPS = ("raw36", "mean3")
```

[Строка 29](../../../../scripts/chromaseed_neural_readout_audit.py#L29)

```python
FAMILIES = ("norm", "perceptual")
```

[Строка 30](../../../../scripts/chromaseed_neural_readout_audit.py#L30)

```python
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)
```

[Строка 39](../../../../scripts/chromaseed_neural_readout_audit.py#L39)

```python
ALPHAS = (0.1, 1.0, 10.0)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `name_for` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_neural_readout_audit.py#L42) |
| `expected_meta` | FunctionDef | См. реализацию | [L47](../../../../scripts/chromaseed_neural_readout_audit.py#L47) |
| `parameter_check` | FunctionDef | См. реализацию | [L103](../../../../scripts/chromaseed_neural_readout_audit.py#L103) |
| `bank_check` | FunctionDef | См. реализацию | [L115](../../../../scripts/chromaseed_neural_readout_audit.py#L115) |
| `main` | FunctionDef | См. реализацию | [L302](../../../../scripts/chromaseed_neural_readout_audit.py#L302) |
