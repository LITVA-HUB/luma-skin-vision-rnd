# `scripts/skin_support_curve_interventions.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_support_curve_interventions.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen-model texture/adapter removal with actual instrument color scoring.

SHA-256 исходника: `f399614f712b037403ec03473e70951a64b2ea5205a17f480030cf1ae973e220`. Строк: **83**.

## Зависимости

```python
import os
import json
from pathlib import Path
import numpy as np
import torch
from skin_support_curve import SkinRepresentation,patient_roles,pixel_patches
from skin_support_curve_train import OUT,RUN,bindings,predict
from skin_pair_train import subset,rows,write
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_audit import scalar_de
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/skin_support_curve_interventions.py#L17)

```python
PROTOCOL=ROOT/'docs/research/skin_support_curve_interventions_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `intervene` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_support_curve_interventions.py#L20) |
| `main` | FunctionDef | См. реализацию | [L30](../../../../scripts/skin_support_curve_interventions.py#L30) |
