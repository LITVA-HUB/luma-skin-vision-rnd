# `scripts/chromaseed_fused_optimizer_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_fused_optimizer_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Small CPU synthetic feasibility probes; no CUDA context and no production data.

SHA-256 исходника: `973b80fd223f272f241157963d2623031c1d3072e6e9ba6ab5da176c17581bea`. Строк: **147**.

## Зависимости

```python
from __future__ import annotations
import importlib
import sys
from pathlib import Path
import numpy as np
import torch
from chromaseed_architecture_scale import Bank
from chromaseed_fused_optimizer import FusedBankAdamW
from chromaseed_refine import BankAdamW
from skin_local_search_train import sha, write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_fused_optimizer_probe.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_fused_optimizer_probe.py#L17)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_fused_optimizer_cpu_probe"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `stream_probe` | FunctionDef | См. реализацию | [L20](../../../../scripts/chromaseed_fused_optimizer_probe.py#L20) |
| `coupled_probe` | FunctionDef | См. реализацию | [L67](../../../../scripts/chromaseed_fused_optimizer_probe.py#L67) |
| `main` | FunctionDef | См. реализацию | [L105](../../../../scripts/chromaseed_fused_optimizer_probe.py#L105) |
