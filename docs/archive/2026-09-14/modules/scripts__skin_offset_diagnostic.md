# `scripts/skin_offset_diagnostic.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_offset_diagnostic.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Privileged source-reference calibration diagnostic; never a production score.

SHA-256 исходника: `3d4f86c03691cb0af691ae5454e3a4fb14bf41c4635e2826f6d40f31f69dcc58`. Строк: **71**.

## Зависимости

```python
import argparse,json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write
from skin_distribution_train import score
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_offset_diagnostic.py#L9)

```python
SOURCE=ROOT/'docs/benchmarks/skin_sampling_transfer_v1'
```

[Строка 9](../../../../scripts/skin_offset_diagnostic.py#L9)

```python
SOURCE_RUN=ROOT/'experiments/runs/skin_sampling_transfer_v1'
```

[Строка 10](../../../../scripts/skin_offset_diagnostic.py#L10)

```python
OUT=ROOT/'docs/benchmarks/skin_offset_diagnostic_v1'
```

[Строка 10](../../../../scripts/skin_offset_diagnostic.py#L10)

```python
RUN=ROOT/'experiments/runs/skin_offset_diagnostic_v1'
```

[Строка 11](../../../../scripts/skin_offset_diagnostic.py#L11)

```python
PROTOCOL=ROOT/'docs/research/skin_offset_diagnostic_protocol_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `excluded_person_offsets` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_offset_diagnostic.py#L14) |
| `bindings` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_offset_diagnostic.py#L31) |
| `main` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_offset_diagnostic.py#L38) |
