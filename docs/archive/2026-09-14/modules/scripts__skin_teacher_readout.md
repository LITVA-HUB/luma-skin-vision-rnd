# `scripts/skin_teacher_readout.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_teacher_readout.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Finite, source-only Ridge screen of licensed frozen teacher information.

SHA-256 исходника: `9865e5f24a403a5669473e64c16a9a3e64e0d17b26bb98377c4fbccedbdd36aa`. Строк: **136**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load as load_pixels
from skin_teacher_features import OUT, CACHE, PROTOCOL, load as load_teacher
from skin_pair_train import subset, rows, write
from skin_mskcc_summary_pilot import summarize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/skin_teacher_readout.py#L15)

```python
RUN = ROOT/'experiments/runs/skin_teacher_readout_v1'
```

[Строка 16](../../../../scripts/skin_teacher_readout.py#L16)

```python
ARMS = ['color', 'teacher', 'combined', 'shuffle17', 'shuffle29', 'shuffle43']
```

[Строка 17](../../../../scripts/skin_teacher_readout.py#L17)

```python
ALPHAS = [.001, .01, .1, 1., 10., 100.]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit_blocks` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_teacher_readout.py#L20) |
| `transform_blocks` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_teacher_readout.py#L29) |
| `site_weights` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_teacher_readout.py#L34) |
| `source_permutation` | FunctionDef | См. реализацию | [L40](../../../../scripts/skin_teacher_readout.py#L40) |
| `blocks` | FunctionDef | См. реализацию | [L47](../../../../scripts/skin_teacher_readout.py#L47) |
| `source_data` | FunctionDef | См. реализацию | [L55](../../../../scripts/skin_teacher_readout.py#L55) |
| `protocol_data` | FunctionDef | См. реализацию | [L64](../../../../scripts/skin_teacher_readout.py#L64) |
| `designs` | FunctionDef | См. реализацию | [L70](../../../../scripts/skin_teacher_readout.py#L70) |
| `main` | FunctionDef | См. реализацию | [L78](../../../../scripts/skin_teacher_readout.py#L78) |
