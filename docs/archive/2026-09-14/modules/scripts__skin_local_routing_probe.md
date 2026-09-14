# `scripts/skin_local_routing_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_routing_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Known-source TRAIN routing test of frozen patch color experts.

SHA-256 исходника: `15604c2508d04d8515549f2e9c1e2e7d61dbaf227bb2787d57faa1217de1bd6c`. Строк: **76**.

## Зависимости

```python
import itertools,json,hashlib
from pathlib import Path
import numpy as np
import torch
from skin_capture_model import CaptureColor,MODES
from skin_capture_support import make_plan,apply_plan
from skin_capture_support_train import OUT as SOURCE,RUN as SOURCE_RUN
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import write
from skin_mskcc_audit import scalar_de
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/skin_local_routing_probe.py#L15)

```python
OUT=ROOT/'docs/benchmarks/skin_capture_support_v1'
```

[Строка 16](../../../../scripts/skin_local_routing_probe.py#L16)

```python
RUN=ROOT/'experiments/runs/skin_capture_support_v1'
```

[Строка 17](../../../../scripts/skin_local_routing_probe.py#L17)

```python
PROTOCOL=ROOT/'docs/research/skin_local_routing_probe_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `route` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_local_routing_probe.py#L20) |
| `main` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_local_routing_probe.py#L26) |
