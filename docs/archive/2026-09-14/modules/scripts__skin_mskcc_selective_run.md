# `scripts/skin_mskcc_selective_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_selective_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched risk fitting -> locked calibration -> locked independent skin test.

SHA-256 исходника: `76e8cee341cbc1f166179fb33ce139ae90dbe5a9664886fc76aa99a1e2703e14`. Строк: **171**.

## Зависимости

```python
import argparse
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,MANIFEST,RAW,sha
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import OUT,RUN,PROTOCOL,designs,deployment_designs,binding,verify_lock
from skin_mskcc_selective_data import sealed
from skin_mskcc_selective_metrics import selection,evaluate,accepted,COVERAGES
from skin_mskcc_color_registry import paths as color_paths,all_predictions
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `write` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_mskcc_selective_run.py#L25) |
| `candidates` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_mskcc_selective_run.py#L29) |
| `score` | FunctionDef | См. реализацию | [L37](../../../../scripts/skin_mskcc_selective_run.py#L37) |
| `fit_heads` | FunctionDef | См. реализацию | [L43](../../../../scripts/skin_mskcc_selective_run.py#L43) |
| `calibrate` | FunctionDef | См. реализацию | [L83](../../../../scripts/skin_mskcc_selective_run.py#L83) |
| `bootstrap_pair` | FunctionDef | См. реализацию | [L109](../../../../scripts/skin_mskcc_selective_run.py#L109) |
| `test` | FunctionDef | См. реализацию | [L120](../../../../scripts/skin_mskcc_selective_run.py#L120) |
| `main` | FunctionDef | См. реализацию | [L161](../../../../scripts/skin_mskcc_selective_run.py#L161) |
