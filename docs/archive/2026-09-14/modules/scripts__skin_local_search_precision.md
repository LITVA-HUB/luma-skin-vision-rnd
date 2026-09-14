# `scripts/skin_local_search_precision.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_search_precision.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Freeze and evaluate storage-only precision variants of final local-search models.

SHA-256 исходника: `f39bede6e8c43b4e0a1ab537292b78d7f0270883ab3ff0c4aaa5d43f7c323e65`. Строк: **383**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import skin_local_search_train as runner
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/skin_local_search_precision.py#L13)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 14](../../../../scripts/skin_local_search_precision.py#L14)

```python
CACHE_HASH = runner.CACHE_HASH
```

[Строка 15](../../../../scripts/skin_local_search_precision.py#L15)

```python
FORMATS = ("fp32_reference", "fp16", "int8")
```

[Строка 16](../../../../scripts/skin_local_search_precision.py#L16)

```python
PROTOCOLS = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 17](../../../../scripts/skin_local_search_precision.py#L17)

```python
METHOD_SEEDS = {
    "ridge": (17,),
    "krr": (17,),
    "random_rbf": (17, 29, 43),
    "guided_rbf": (17, 29, 43),
    "mlp": (17, 29, 43),
}
```

[Строка 24](../../../../scripts/skin_local_search_precision.py#L24)

```python
SCALER_KEYS = frozenset(("x_mean", "x_std", "y_mean", "y_std"))
```

[Строка 25](../../../../scripts/skin_local_search_precision.py#L25)

```python
MODEL_NUMERIC_KEYS = {
    "ridge": frozenset((*SCALER_KEYS, "beta")),
    "krr": frozenset((*SCALER_KEYS, "centers", "widths", "beta")),
    "random_rbf": frozenset((*SCALER_KEYS, "centers", "widths", "beta")),
    "guided_rbf": frozenset((*SCALER_KEYS, "centers", "widths", "beta")),
    "mlp": frozenset(
        (*SCALER_KEYS, "hidden_w", "hidden_b", "out_w", "out_b", "skip_w")
    ),
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `sha256` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_local_search_precision.py#L36) |
| `load_npz` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_local_search_precision.py#L44) |
| `_method` | FunctionDef | См. реализацию | [L49](../../../../scripts/skin_local_search_precision.py#L49) |
| `_validate_source` | FunctionDef | См. реализацию | [L59](../../../../scripts/skin_local_search_precision.py#L59) |
| `_quantize_symmetric` | FunctionDef | См. реализацию | [L77](../../../../scripts/skin_local_search_precision.py#L77) |
| `_int8_rule` | FunctionDef | См. реализацию | [L96](../../../../scripts/skin_local_search_precision.py#L96) |
| `pack_model` | FunctionDef | Convert one canonical FP32 model to a fixed storage representation. | [L106](../../../../scripts/skin_local_search_precision.py#L106) |
| `dequantize_model` | FunctionDef | Materialize an FP32 inference model from one stored representation. | [L133](../../../../scripts/skin_local_search_precision.py#L133) |
| `payload_stats` | FunctionDef | См. реализацию | [L164](../../../../scripts/skin_local_search_precision.py#L164) |
| `_expected_sources` | FunctionDef | См. реализацию | [L178](../../../../scripts/skin_local_search_precision.py#L178) |
| `_same_payload` | FunctionDef | См. реализацию | [L192](../../../../scripts/skin_local_search_precision.py#L192) |
| `_save_payload_once` | FunctionDef | См. реализацию | [L196](../../../../scripts/skin_local_search_precision.py#L196) |
| `_json_bytes` | FunctionDef | См. реализацию | [L205](../../../../scripts/skin_local_search_precision.py#L205) |
| `_write_once` | FunctionDef | См. реализацию | [L209](../../../../scripts/skin_local_search_precision.py#L209) |
| `_verify_manifest` | FunctionDef | См. реализацию | [L220](../../../../scripts/skin_local_search_precision.py#L220) |
| `prepare` | FunctionDef | Freeze three storage formats without opening any data or evaluation arrays. | [L244](../../../../scripts/skin_local_search_precision.py#L244) |
| `_deviation` | FunctionDef | См. реализацию | [L296](../../../../scripts/skin_local_search_precision.py#L296) |
| `evaluate` | FunctionDef | Evaluate every frozen format on all outer roles, without choosing a format. | [L311](../../../../scripts/skin_local_search_precision.py#L311) |
| `main` | FunctionDef | См. реализацию | [L364](../../../../scripts/skin_local_search_precision.py#L364) |
