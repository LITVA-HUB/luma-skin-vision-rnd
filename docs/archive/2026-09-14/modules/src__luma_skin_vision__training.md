# `src/luma_skin_vision/training.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/training.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Local experiment lifecycle. Frozen manifest binding and separate fit roles.

SHA-256 исходника: `c1f30e5a6f4653f677e54e073076cef4b8175a277dc38b1086c8e03ce75f0a64`. Строк: **499**.

## Зависимости

```python
import json
import random
import time
from pathlib import Path
import numpy as np
from luma_skin_vision.calibration import Calibrator
from luma_skin_vision.color import (
    delta_e00,
    lab_to_srgb,
    linear_to_srgb,
    srgb_to_lab,
    srgb_to_linear,
)
from luma_skin_vision.data import dataset_identity, sha256, validate_records
from luma_skin_vision.environment import inspect_environment
from luma_skin_vision.evaluation import risk_coverage, summarize
from luma_skin_vision.experiment import create_run, write_json
from luma_skin_vision.gates import measurement_gate
from luma_skin_vision.photometry import apply_ccm, fit_ccm, hypotheses
from luma_skin_vision.preprocessing import decode_image, region_tensor
from luma_skin_vision.roi import ambiguity_features, cheek_masks, robust_rgb
from luma_skin_vision.uncertainty import error_features, fit_error, predict_error
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 28](../../../../src/luma_skin_vision/training.py#L28)

```python
LEARNED = ("baseline_c", "baseline_c_plus", "proposed_v1")
```

[Строка 29](../../../../src/luma_skin_vision/training.py#L29)

```python
METHODS = ("baseline_a0", "baseline_a1", "baseline_a2") + LEARNED
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `validate_config` | FunctionDef | См. реализацию | [L32](../../../../src/luma_skin_vision/training.py#L32) |
| `prepare` | FunctionDef | См. реализацию | [L65](../../../../src/luma_skin_vision/training.py#L65) |
| `_seed` | FunctionDef | См. реализацию | [L91](../../../../src/luma_skin_vision/training.py#L91) |
| `_device` | FunctionDef | См. реализацию | [L104](../../../../src/luma_skin_vision/training.py#L104) |
| `_fit_model` | FunctionDef | См. реализацию | [L117](../../../../src/luma_skin_vision/training.py#L117) |
| `_predict_model` | FunctionDef | См. реализацию | [L190](../../../../src/luma_skin_vision/training.py#L190) |
| `_fit_deterministic` | FunctionDef | См. реализацию | [L205](../../../../src/luma_skin_vision/training.py#L205) |
| `_predict_deterministic` | FunctionDef | См. реализацию | [L220](../../../../src/luma_skin_vision/training.py#L220) |
| `train` | FunctionDef | См. реализацию | [L229](../../../../src/luma_skin_vision/training.py#L229) |
| `load_run` | FunctionDef | См. реализацию | [L347](../../../../src/luma_skin_vision/training.py#L347) |
| `predictions` | FunctionDef | См. реализацию | [L372](../../../../src/luma_skin_vision/training.py#L372) |
| `calibrate_run` | FunctionDef | См. реализацию | [L395](../../../../src/luma_skin_vision/training.py#L395) |
| `evaluate_run` | FunctionDef | См. реализацию | [L411](../../../../src/luma_skin_vision/training.py#L411) |
