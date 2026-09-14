# `scripts/skin_correction_transfer.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_correction_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Unchanged stable-color correction recipes on camera-held-out source banks.

SHA-256 исходника: `f70ab190bf6e2482307ba96ebce7041b1e0de4d06214f01bb73dde985ea15eeb`. Строк: **189**.

## Зависимости

```python
import os
import argparse,json
from pathlib import Path
import numpy as np
import torch
from skin_crossfit_correction import ARMS,features,StableColorModel
from skin_crossfit_correction_train import fit_core,fit_head,predict,score,table_scores
from skin_support_curve import SkinRepresentation
from skin_local_reference_transfer import banks
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/skin_correction_transfer.py#L17)

```python
OUT=ROOT/'docs/benchmarks/skin_correction_transfer_v1'
```

[Строка 17](../../../../scripts/skin_correction_transfer.py#L17)

```python
RUN=ROOT/'experiments/runs/skin_correction_transfer_v1'
```

[Строка 18](../../../../scripts/skin_correction_transfer.py#L18)

```python
SOURCE=ROOT/'experiments/runs/skin_sampling_transfer_v1'
```

[Строка 18](../../../../scripts/skin_correction_transfer.py#L18)

```python
PROTOCOL=ROOT/'docs/research/skin_correction_transfer_protocol_v1.md'
```

[Строка 19](../../../../scripts/skin_correction_transfer.py#L19)

```python
SEEDS=(17,29,43)
```

[Строка 20](../../../../scripts/skin_correction_transfer.py#L20)

```python
RECOVERY=ROOT/'docs/research/skin_correction_transfer_batch_recovery.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `four_folds` | FunctionDef | См. реализацию | [L23](../../../../scripts/skin_correction_transfer.py#L23) |
| `route_tables` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_correction_transfer.py#L36) |
| `source_file` | FunctionDef | См. реализацию | [L41](../../../../scripts/skin_correction_transfer.py#L41) |
| `predict_domains` | FunctionDef | См. реализацию | [L44](../../../../scripts/skin_correction_transfer.py#L44) |
| `active_lock` | FunctionDef | См. реализацию | [L56](../../../../scripts/skin_correction_transfer.py#L56) |
| `bindings` | FunctionDef | См. реализацию | [L59](../../../../scripts/skin_correction_transfer.py#L59) |
| `score_domains` | FunctionDef | См. реализацию | [L71](../../../../scripts/skin_correction_transfer.py#L71) |
| `write_arrays` | FunctionDef | См. реализацию | [L75](../../../../scripts/skin_correction_transfer.py#L75) |
| `run_bank` | FunctionDef | См. реализацию | [L82](../../../../scripts/skin_correction_transfer.py#L82) |
| `main` | FunctionDef | См. реализацию | [L166](../../../../scripts/skin_correction_transfer.py#L166) |
