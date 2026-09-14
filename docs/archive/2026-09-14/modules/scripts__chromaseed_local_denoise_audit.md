# `scripts/chromaseed_local_denoise_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_local_denoise_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent ND payload, inner selection, stage and actual-consumer audit.

SHA-256 исходника: `49fa9faa6ae1b15819da798df655a3cb71d2aad4cb49d33f404e84d4e1ddfd52`. Строк: **385**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer
from chromaseed_gaussian_audit import direct as reference_direct
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_numpy import Predictor
from chromaseed_local_denoise_train import CACHE, ROOT, RUN, load_data
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/chromaseed_local_denoise_audit.py#L20)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_local_denoise_v1"
```

[Строка 21](../../../../scripts/chromaseed_local_denoise_audit.py#L21)

```python
FAMILIES = ("plain", "local2", "local4", "blind4", "e2e4")
```

[Строка 22](../../../../scripts/chromaseed_local_denoise_audit.py#L22)

```python
SEEDS = (17, 29, 43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `close` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_local_denoise_audit.py#L25) |
| `direct` | FunctionDef | Separate neuron reduction and algebraic recurrence; no ND forward helper. | [L40](../../../../scripts/chromaseed_local_denoise_audit.py#L40) |
| `verify_normalizers` | FunctionDef | См. реализацию | [L76](../../../../scripts/chromaseed_local_denoise_audit.py#L76) |
| `full_metrics` | FunctionDef | См. реализацию | [L84](../../../../scripts/chromaseed_local_denoise_audit.py#L84) |
| `audit_inner` | FunctionDef | См. реализацию | [L98](../../../../scripts/chromaseed_local_denoise_audit.py#L98) |
| `audit_final` | FunctionDef | См. реализацию | [L186](../../../../scripts/chromaseed_local_denoise_audit.py#L186) |
| `paired` | FunctionDef | См. реализацию | [L297](../../../../scripts/chromaseed_local_denoise_audit.py#L297) |
| `main` | FunctionDef | См. реализацию | [L330](../../../../scripts/chromaseed_local_denoise_audit.py#L330) |
