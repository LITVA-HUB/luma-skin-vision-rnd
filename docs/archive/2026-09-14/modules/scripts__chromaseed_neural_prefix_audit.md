# `scripts/chromaseed_neural_prefix_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_prefix_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent NP pruning, conditional selection and all-consumer audit.

SHA-256 исходника: `852a6b354945e3c1e93f684efe71c05f3b515d3110699332b043581f3b0b4949`. Строк: **322**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, direct, full_metrics, verify_normalizers
from chromaseed_neural_prefix_numpy import Predictor, export_prefix, predict
from chromaseed_neural_prefix_run import ND, ND_OUT, OUT, ROOT, RUN, check_map, load_data
from chromaseed_refine_audit import error_summary
from skin_local_search_train import folds_for, roles, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_neural_prefix_audit.py#L19)

```python
SIZES = {
    "plain": (1, 36, 69),
    "local2": (2, 39, 32),
    "local4": (4, 39, 16),
    "blind4": (4, 39, 16),
    "e2e4": (4, 39, 16),
}
```

[Строка 26](../../../../scripts/chromaseed_neural_prefix_audit.py#L26)

```python
SEEDS = (17, 29, 43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `inspect_export` | FunctionDef | Account for every retained array without calling the NP exporter. | [L29](../../../../scripts/chromaseed_neural_prefix_audit.py#L29) |
| `inner` | FunctionDef | См. реализацию | [L55](../../../../scripts/chromaseed_neural_prefix_audit.py#L55) |
| `final` | FunctionDef | См. реализацию | [L153](../../../../scripts/chromaseed_neural_prefix_audit.py#L153) |
| `paired` | FunctionDef | См. реализацию | [L236](../../../../scripts/chromaseed_neural_prefix_audit.py#L236) |
| `main` | FunctionDef | См. реализацию | [L273](../../../../scripts/chromaseed_neural_prefix_audit.py#L273) |
