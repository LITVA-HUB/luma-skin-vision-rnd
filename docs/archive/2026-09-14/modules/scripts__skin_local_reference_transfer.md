# `scripts/skin_local_reference_transfer.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_reference_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Unchanged six local-reference systems on explicit source camera banks.

SHA-256 исходника: `9f9d7e1c3a8fae24acedd03f74cd65a60bdf5c39e835e2adc07b986d48bcc16a`. Строк: **65**.

## Зависимости

```python
import argparse,json
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_local_reference import METHODS,predict_bank
from skin_local_reference_run import summarize,OUT as SCREEN
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import write,subset
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/skin_local_reference_transfer.py#L12)

```python
OUT=ROOT/'docs/benchmarks/skin_local_reference_transfer_v1'
```

[Строка 12](../../../../scripts/skin_local_reference_transfer.py#L12)

```python
RUN=ROOT/'experiments/runs/skin_local_reference_transfer_v1'
```

[Строка 13](../../../../scripts/skin_local_reference_transfer.py#L13)

```python
PROTOCOL=ROOT/'docs/research/skin_local_reference_transfer_protocol_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bindings` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_local_reference_transfer.py#L16) |
| `banks` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_local_reference_transfer.py#L24) |
| `main` | FunctionDef | См. реализацию | [L33](../../../../scripts/skin_local_reference_transfer.py#L33) |
