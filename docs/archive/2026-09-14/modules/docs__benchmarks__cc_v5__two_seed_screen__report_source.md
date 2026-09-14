# `docs/benchmarks/cc_v5/two_seed_screen/report_source.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/benchmarks/cc_v5/two_seed_screen/report_source.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independently rescore completed V5 runs; no training or image decoding.

SHA-256 исходника: `6014ea59317464f692e449f6a604e6ba2634543e5059991f9d9cd477dc4024d8`. Строк: **180**.

## Зависимости

```python
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../docs/benchmarks/cc_v5/two_seed_screen/report_source.py#L14)

```python
ARMS = ('point', 'posterior_random', 'action_random', 'transport_random',
        'transport_policy', 'transport_gradient')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `reproduction_degrees` | FunctionDef | См. реализацию | [L18](../../../../docs/benchmarks/cc_v5/two_seed_screen/report_source.py#L18) |
| `summarize` | FunctionDef | См. реализацию | [L30](../../../../docs/benchmarks/cc_v5/two_seed_screen/report_source.py#L30) |
| `selective` | FunctionDef | См. реализацию | [L42](../../../../docs/benchmarks/cc_v5/two_seed_screen/report_source.py#L42) |
| `verify_manifest` | FunctionDef | См. реализацию | [L55](../../../../docs/benchmarks/cc_v5/two_seed_screen/report_source.py#L55) |
| `generate` | FunctionDef | См. реализацию | [L65](../../../../docs/benchmarks/cc_v5/two_seed_screen/report_source.py#L65) |
