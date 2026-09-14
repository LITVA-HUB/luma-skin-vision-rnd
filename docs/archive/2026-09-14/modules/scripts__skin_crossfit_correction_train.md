# `scripts/skin_crossfit_correction_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_crossfit_correction_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Nine person-excluded core fits and nine matched stable-color correction heads.

SHA-256 исходника: `1fe4a05021e9cea1c2aab079a470ce3f1947231f33c7f3bf8dd94bbf7fa29471`. Строк: **178**.

## Зависимости

```python
import os
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_crossfit_correction import ARMS,inner_folds,choose_predictions,features,StableHead,StableColorModel
from skin_support_curve import SkinRepresentation,patient_roles
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,write
from skin_local_reference_run import summarize
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/skin_crossfit_correction_train.py#L18)

```python
OUT=ROOT/'docs/benchmarks/skin_crossfit_correction_v1'
```

[Строка 18](../../../../scripts/skin_crossfit_correction_train.py#L18)

```python
RUN=ROOT/'experiments/runs/skin_crossfit_correction_v1'
```

[Строка 19](../../../../scripts/skin_crossfit_correction_train.py#L19)

```python
SOURCE=ROOT/'experiments/runs/skin_support_curve_v1'
```

[Строка 19](../../../../scripts/skin_crossfit_correction_train.py#L19)

```python
PROTOCOL=ROOT/'docs/research/skin_crossfit_correction_protocol_v1.md'
```

[Строка 20](../../../../scripts/skin_crossfit_correction_train.py#L20)

```python
SEEDS=(17,29,43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `source_file` | FunctionDef | См. реализацию | [L23](../../../../scripts/skin_crossfit_correction_train.py#L23) |
| `bindings` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_crossfit_correction_train.py#L26) |
| `predict` | FunctionDef | См. реализацию | [L35](../../../../scripts/skin_crossfit_correction_train.py#L35) |
| `fit_core` | FunctionDef | См. реализацию | [L40](../../../../scripts/skin_crossfit_correction_train.py#L40) |
| `table_scores` | FunctionDef | См. реализацию | [L65](../../../../scripts/skin_crossfit_correction_train.py#L65) |
| `score` | FunctionDef | См. реализацию | [L68](../../../../scripts/skin_crossfit_correction_train.py#L68) |
| `fit_head` | FunctionDef | См. реализацию | [L78](../../../../scripts/skin_crossfit_correction_train.py#L78) |
| `run_seed` | FunctionDef | См. реализацию | [L96](../../../../scripts/skin_crossfit_correction_train.py#L96) |
| `main` | FunctionDef | См. реализацию | [L165](../../../../scripts/skin_crossfit_correction_train.py#L165) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L35–37</summary>

```python
def predict(model,tokens,mean,std):
    model.eval()
    with torch.no_grad():return np.concatenate([(model(t,None)[0]*std+mean).cpu().numpy() for t in tokens.split(32)])
```

</details>
