# `scripts/skin_gradient_transfer_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_gradient_transfer_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen source checkpoint derivative and finite-update diagnostic.

SHA-256 исходника: `4d66c61b27de334494b8c59e728e6e203db97af5156056866fd6fff78ea38834`. Строк: **135**.

## Зависимости

```python
import os
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_gradient_transfer import partitions,group_means,gradient_statistics,transient_step,component_gradients,STEPS
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_support_curve import SkinRepresentation
from skin_capture_model import MODES
from skin_pair_train import subset,write
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/skin_gradient_transfer_run.py#L17)

```python
OUT=ROOT/'docs/benchmarks/skin_gradient_transfer_v1'
```

[Строка 18](../../../../scripts/skin_gradient_transfer_run.py#L18)

```python
RUN=ROOT/'experiments/runs/skin_gradient_transfer_v1'
```

[Строка 19](../../../../scripts/skin_gradient_transfer_run.py#L19)

```python
SOURCE=ROOT/'experiments/runs/skin_sampling_transfer_v1'
```

[Строка 20](../../../../scripts/skin_gradient_transfer_run.py#L20)

```python
PROTOCOL=ROOT/'docs/research/skin_gradient_transfer_protocol_v1.md'
```

[Строка 21](../../../../scripts/skin_gradient_transfer_run.py#L21)

```python
MODELS=[f'{p}__{a}__s17' for p in ('mixed','from_SLR','from_ipod') for a in ('image','person_color')]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bindings` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_gradient_transfer_run.py#L24) |
| `setup` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_gradient_transfer_run.py#L34) |
| `values` | FunctionDef | См. реализацию | [L48](../../../../scripts/skin_gradient_transfer_run.py#L48) |
| `run_model` | FunctionDef | См. реализацию | [L58](../../../../scripts/skin_gradient_transfer_run.py#L58) |
| `main` | FunctionDef | См. реализацию | [L120](../../../../scripts/skin_gradient_transfer_run.py#L120) |
