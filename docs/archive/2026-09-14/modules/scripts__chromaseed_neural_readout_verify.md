# `scripts/chromaseed_neural_readout_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_readout_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Seal NR once; validate existing NR/TG/FG/C/H/X/A receipts without rewriting.

SHA-256 исходника: `a2a6cf887fdc442d786af1e3f9d44e5b8d8b6fbac4425642e30ac0cd83068d2c`. Строк: **501**.

## Зависимости

```python
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from chromaseed_gate_stability_audit import close
from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/chromaseed_neural_readout_verify.py#L20)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 21](../../../../scripts/chromaseed_neural_readout_verify.py#L21)

```python
RUN = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
```

[Строка 22](../../../../scripts/chromaseed_neural_readout_verify.py#L22)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_neural_readout_v1"
```

[Строка 23](../../../../scripts/chromaseed_neural_readout_verify.py#L23)

```python
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-readout-2026-09-13.md"
```

[Строка 24](../../../../scripts/chromaseed_neural_readout_verify.py#L24)

```python
SOURCE = "70246d9935e52d5be0dcebf758ba3e63b30be4965a9154a19ba180f5cf1782c1"
```

[Строка 25](../../../../scripts/chromaseed_neural_readout_verify.py#L25)

```python
SELECTION = "258ba273abff66fbac6f442736f9af0ec3312b204f5cbc9e9310d6ecf4154954"
```

[Строка 26](../../../../scripts/chromaseed_neural_readout_verify.py#L26)

```python
RESULT = "b13dbb2cfde6a8e85dbe4d35f4d5254cd384c916c5b5d2a323c1d0e1ecf67522"
```

[Строка 27](../../../../scripts/chromaseed_neural_readout_verify.py#L27)

```python
TG_RECEIPT = "19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `matched` | FunctionDef | См. реализацию | [L30](../../../../scripts/chromaseed_neural_readout_verify.py#L30) |
| `aggregate_checks` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_neural_readout_verify.py#L34) |
| `main` | FunctionDef | См. реализацию | [L185](../../../../scripts/chromaseed_neural_readout_verify.py#L185) |
