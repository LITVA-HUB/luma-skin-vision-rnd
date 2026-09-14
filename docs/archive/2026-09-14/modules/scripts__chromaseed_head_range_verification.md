# `scripts/chromaseed_head_range_verification.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_head_range_verification.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent HR contracts, terminal guards, selection and cost accounting.

SHA-256 исходника: `0eab3d471b2d2a76a8bd8355f2f2aad87681f70aa1af5f5e3cdae40a8f27d44d`. Строк: **382**.

## Зависимости

```python
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../scripts/chromaseed_head_range_verification.py#L16)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 17](../../../../scripts/chromaseed_head_range_verification.py#L17)

```python
RUN = ROOT / "experiments/runs/chromaseed_head_range_v1"
```

[Строка 18](../../../../scripts/chromaseed_head_range_verification.py#L18)

```python
OUT = ROOT / "docs/benchmarks/chromaseed_head_range_v1"
```

[Строка 19](../../../../scripts/chromaseed_head_range_verification.py#L19)

```python
CONTRACT = RUN / "verification_v1/protocol.json"
```

[Строка 20](../../../../scripts/chromaseed_head_range_verification.py#L20)

```python
AS_RUN = ROOT / "experiments/runs/chromaseed_architecture_scale_v1"
```

[Строка 21](../../../../scripts/chromaseed_head_range_verification.py#L21)

```python
AS_OUT = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1"
```

[Строка 22](../../../../scripts/chromaseed_head_range_verification.py#L22)

```python
SOURCE = "dfb96f902658009f57d37478ee9e2964c951a2bbce8aaba9acda77ff142652b2"
```

[Строка 23](../../../../scripts/chromaseed_head_range_verification.py#L23)

```python
AS_SEAL = "ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9"
```

[Строка 24](../../../../scripts/chromaseed_head_range_verification.py#L24)

```python
RECOVERY = "5952cf01a7a77db9f8945fed32f09294d5002d62a8d9dee5953c41308c3f637b"
```

[Строка 25](../../../../scripts/chromaseed_head_range_verification.py#L25)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 26](../../../../scripts/chromaseed_head_range_verification.py#L26)

```python
MODES = ("unit", "wide", "linear")
```

[Строка 27](../../../../scripts/chromaseed_head_range_verification.py#L27)

```python
SEEDS = (17, 29, 43)
```

[Строка 28](../../../../scripts/chromaseed_head_range_verification.py#L28)

```python
RATES = (1e-5, 1e-4)
```

[Строка 29](../../../../scripts/chromaseed_head_range_verification.py#L29)

```python
TIMES = (128, 512, 2048)
```

[Строка 30](../../../../scripts/chromaseed_head_range_verification.py#L30)

```python
PARAMETERS = dict(
    patch_small=17374,
    patch5m=4962566,
    soft_small=15246,
    soft5m=4846822,
    dynamic_small=15246,
    dynamic5m=4846822,
    pool5m=4851846,
)
```

[Строка 39](../../../../scripts/chromaseed_head_range_verification.py#L39)

```python
FILES = (
    "scripts/chromaseed_head_range_verification.py",
    "scripts/chromaseed_head_range_audit.py",
    "scripts/chromaseed_head_range_runtime.py",
    "scripts/chromaseed_head_range_report.py",
    "tests/test_chromaseed_head_range_verification.py",
    "docs/research/chromaseed_head_range_verification_v1_protocol.md",
)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_head_range_verification.py#L49) |
| `read` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_head_range_verification.py#L54) |
| `write_once` | FunctionDef | Never replace an existing receipt, including one from an interrupted stage. | [L58](../../../../scripts/chromaseed_head_range_verification.py#L58) |
| `check_hashes` | FunctionDef | См. реализацию | [L71](../../../../scripts/chromaseed_head_range_verification.py#L71) |
| `remember` | FunctionDef | См. реализацию | [L77](../../../../scripts/chromaseed_head_range_verification.py#L77) |
| `process_alive` | FunctionDef | True/False/unknown; permission errors never count as a dead worker. | [L89](../../../../scripts/chromaseed_head_range_verification.py#L89) |
| `require_terminal` | FunctionDef | См. реализацию | [L118](../../../../scripts/chromaseed_head_range_verification.py#L118) |
| `competing_workers` | FunctionDef | См. реализацию | [L130](../../../../scripts/chromaseed_head_range_verification.py#L130) |
| `require_quiet_host` | FunctionDef | Fail closed before benchmarks/replay when a competing workload is observed. | [L155](../../../../scripts/chromaseed_head_range_verification.py#L155) |
| `compare` | FunctionDef | См. реализацию | [L193](../../../../scripts/chromaseed_head_range_verification.py#L193) |
| `rank` | FunctionDef | См. реализацию | [L201](../../../../scripts/chromaseed_head_range_verification.py#L201) |
| `select_policies` | FunctionDef | См. реализацию | [L214](../../../../scripts/chromaseed_head_range_verification.py#L214) |
| `index_records` | FunctionDef | См. реализацию | [L229](../../../../scripts/chromaseed_head_range_verification.py#L229) |
| `operational_costs` | FunctionDef | См. реализацию | [L239](../../../../scripts/chromaseed_head_range_verification.py#L239) |
| `recovery_bindings` | FunctionDef | См. реализацию | [L268](../../../../scripts/chromaseed_head_range_verification.py#L268) |
| `freeze_contract` | FunctionDef | См. реализацию | [L302](../../../../scripts/chromaseed_head_range_verification.py#L302) |
| `verify_contract` | FunctionDef | См. реализацию | [L346](../../../../scripts/chromaseed_head_range_verification.py#L346) |
| `verify_stage` | FunctionDef | См. реализацию | [L357](../../../../scripts/chromaseed_head_range_verification.py#L357) |
| `main` | FunctionDef | См. реализацию | [L366](../../../../scripts/chromaseed_head_range_verification.py#L366) |
