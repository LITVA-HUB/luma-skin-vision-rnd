# `scripts/skin_relational_probe_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_relational_probe_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

TRAIN same-site/color-hard controls and48 excluded-person linear fits.

SHA-256 исходника: `ba5095e23bf9305616a0c11d778042d74efb118da05a079b715c509959a00548`. Строк: **130**.

## Зависимости

```python
import os
import argparse,itertools,json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from luma_skin_vision.color import delta_e00
from skin_relational_probe import matched_controls,loo_ridge,excluded_standardize,paired_preference,anchor_average,complete_potential
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_gradient_transfer_run import setup,SOURCE
from skin_pair_train import write
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/skin_relational_probe_run.py#L17)

```python
OUT=ROOT/'docs/benchmarks/skin_relational_probe_v1'
```

[Строка 17](../../../../scripts/skin_relational_probe_run.py#L17)

```python
RUN=ROOT/'experiments/runs/skin_relational_probe_v1'
```

[Строка 18](../../../../scripts/skin_relational_probe_run.py#L18)

```python
PROTOCOL=ROOT/'docs/research/skin_relational_probe_protocol_v1.md'
```

[Строка 19](../../../../scripts/skin_relational_probe_run.py#L19)

```python
MODELS=('mixed__image__s17','mixed__person_color__s17')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bindings` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_relational_probe_run.py#L22) |
| `context_features` | FunctionDef | См. реализацию | [L32](../../../../scripts/skin_relational_probe_run.py#L32) |
| `distance` | FunctionDef | См. реализацию | [L43](../../../../scripts/skin_relational_probe_run.py#L43) |
| `color_summary` | FunctionDef | См. реализацию | [L49](../../../../scripts/skin_relational_probe_run.py#L49) |
| `correlation` | FunctionDef | См. реализацию | [L55](../../../../scripts/skin_relational_probe_run.py#L55) |
| `main` | FunctionDef | См. реализацию | [L60](../../../../scripts/skin_relational_probe_run.py#L60) |
