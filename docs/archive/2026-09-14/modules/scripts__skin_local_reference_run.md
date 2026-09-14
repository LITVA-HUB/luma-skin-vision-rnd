# `scripts/skin_local_reference_run.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_reference_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Six frozen systems on original-TRAIN leave-one-person-out skin color.

SHA-256 исходника: `73784ed7d20d4fd05bf47df399148fc3d92380f6c91eca7b1ba76f143ca39284`. Строк: **78**.

## Зависимости

```python
import argparse,json,time
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_local_reference import METHODS,predict_bank
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import write
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/skin_local_reference_run.py#L11)

```python
OUT=ROOT/'docs/benchmarks/skin_local_reference_v1'
```

[Строка 11](../../../../scripts/skin_local_reference_run.py#L11)

```python
RUN=ROOT/'experiments/runs/skin_local_reference_v1'
```

[Строка 12](../../../../scripts/skin_local_reference_run.py#L12)

```python
PROTOCOL=ROOT/'docs/research/skin_local_reference_protocol_v1.md'
```

[Строка 13](../../../../scripts/skin_local_reference_run.py#L13)

```python
PREVIOUS=ROOT/'experiments/runs/skin_relational_probe_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bindings` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_local_reference_run.py#L16) |
| `summarize` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_local_reference_run.py#L25) |
| `main` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_local_reference_run.py#L31) |
