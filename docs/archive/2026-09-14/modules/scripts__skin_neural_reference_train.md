# `scripts/skin_neural_reference_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_neural_reference_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen-core, equal-capacity local-reference skin color screen, final fits only.

SHA-256 исходника: `526c139a8884dfcd86ef7cf04da6cae19a24e0dad534c81f34390770fd5b1e2c`. Строк: **203**.

## Зависимости

```python
import os
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_neural_reference import ColorAdapter,DeployedColor,core_features,ARMS
from skin_capture_model import CaptureColor
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_pair_train import write,subset
from skin_local_reference_transfer import banks
from skin_local_reference_run import summarize
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/skin_neural_reference_train.py#L18)

```python
OUT=ROOT/'docs/benchmarks/skin_neural_reference_v1'
```

[Строка 19](../../../../scripts/skin_neural_reference_train.py#L19)

```python
RUN=ROOT/'experiments/runs/skin_neural_reference_v1'
```

[Строка 20](../../../../scripts/skin_neural_reference_train.py#L20)

```python
SOURCE=ROOT/'experiments/runs/skin_capture_v1'
```

[Строка 21](../../../../scripts/skin_neural_reference_train.py#L21)

```python
PROTOCOL=ROOT/'docs/research/skin_neural_reference_protocol_v1.md'
```

[Строка 22](../../../../scripts/skin_neural_reference_train.py#L22)

```python
SEEDS=(17,29,43)
```

[Строка 23](../../../../scripts/skin_neural_reference_train.py#L23)

```python
STEPS=300
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `source_file` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_neural_reference_train.py#L26) |
| `bindings` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_neural_reference_train.py#L29) |
| `features` | FunctionDef | См. реализацию | [L41](../../../../scripts/skin_neural_reference_train.py#L41) |
| `prepare_core` | FunctionDef | См. реализацию | [L49](../../../../scripts/skin_neural_reference_train.py#L49) |
| `risk_order` | FunctionDef | См. реализацию | [L72](../../../../scripts/skin_neural_reference_train.py#L72) |
| `score_arrays` | FunctionDef | См. реализацию | [L77](../../../../scripts/skin_neural_reference_train.py#L77) |
| `deployment` | FunctionDef | См. реализацию | [L93](../../../../scripts/skin_neural_reference_train.py#L93) |
| `profile` | FunctionDef | См. реализацию | [L104](../../../../scripts/skin_neural_reference_train.py#L104) |
| `run_bank` | FunctionDef | См. реализацию | [L121](../../../../scripts/skin_neural_reference_train.py#L121) |
| `main` | FunctionDef | См. реализацию | [L189](../../../../scripts/skin_neural_reference_train.py#L189) |
