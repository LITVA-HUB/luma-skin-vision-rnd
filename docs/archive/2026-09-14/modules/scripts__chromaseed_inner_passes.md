# `scripts/chromaseed_inner_passes.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_inner_passes.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

TRAIN-only AS INNER pass diagnostics; no training or adopted exit policy.

SHA-256 исходника: `fb6a6638cd13bf6f24ee3edea7300da1ef02a689f91f511bdb8bbd3286498cf1`. Строк: **311**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from chromaseed_recurrent_stream import StreamingPredictor
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/chromaseed_inner_passes.py#L15)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 19](../../../../scripts/chromaseed_inner_passes.py#L19)

```python
OUT = Path("D:/Luma-RnD/chromaseed_inner_passes_v1")
```

[Строка 20](../../../../scripts/chromaseed_inner_passes.py#L20)

```python
AS = ROOT / "experiments/runs/chromaseed_architecture_scale_v1"
```

[Строка 21](../../../../scripts/chromaseed_inner_passes.py#L21)

```python
SEAL = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1/verification.json"
```

[Строка 22](../../../../scripts/chromaseed_inner_passes.py#L22)

```python
SEAL_SHA = "ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9"
```

[Строка 23](../../../../scripts/chromaseed_inner_passes.py#L23)

```python
CACHE = Path("C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/processed/skin_mskcc_pixels_v1/train.npz")
```

[Строка 24](../../../../scripts/chromaseed_inner_passes.py#L24)

```python
CACHE_SHA = "d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0"
```

[Строка 25](../../../../scripts/chromaseed_inner_passes.py#L25)

```python
STREAM = Path("D:/Luma-RnD/chromaseed_recurrent_stream_v1/qualification.json")
```

[Строка 26](../../../../scripts/chromaseed_inner_passes.py#L26)

```python
STREAM_SHA = "712520d394f845565d4f6e678070ffdfe04341933b22efb9eb81229ec7eb3fb0"
```

[Строка 27](../../../../scripts/chromaseed_inner_passes.py#L27)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 28](../../../../scripts/chromaseed_inner_passes.py#L28)

```python
VARIANTS = ("soft_small", "dynamic_small", "soft5m", "dynamic5m")
```

[Строка 29](../../../../scripts/chromaseed_inner_passes.py#L29)

```python
SEEDS = (17, 29, 43)
```

[Строка 30](../../../../scripts/chromaseed_inner_passes.py#L30)

```python
RATES = (1e-5, 1e-4)
```

[Строка 31](../../../../scripts/chromaseed_inner_passes.py#L31)

```python
THRESHOLDS = (0.1, 0.25, 0.5, 1.0, 2.0)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_inner_passes.py#L34) |
| `read` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_inner_passes.py#L39) |
| `write_once` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_inner_passes.py#L43) |
| `check_bindings` | FunctionDef | См. реализацию | [L55](../../../../scripts/chromaseed_inner_passes.py#L55) |
| `check_partition` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_inner_passes.py#L61) |
| `summarize` | FunctionDef | См. реализацию | [L75](../../../../scripts/chromaseed_inner_passes.py#L75) |
| `register` | FunctionDef | См. реализацию | [L119](../../../../scripts/chromaseed_inner_passes.py#L119) |
| `load_data` | FunctionDef | См. реализацию | [L168](../../../../scripts/chromaseed_inner_passes.py#L168) |
| `case_rows` | FunctionDef | См. реализацию | [L179](../../../../scripts/chromaseed_inner_passes.py#L179) |
| `saved_oof` | FunctionDef | См. реализацию | [L193](../../../../scripts/chromaseed_inner_passes.py#L193) |
| `infer_case` | FunctionDef | См. реализацию | [L201](../../../../scripts/chromaseed_inner_passes.py#L201) |
| `run` | FunctionDef | См. реализацию | [L238](../../../../scripts/chromaseed_inner_passes.py#L238) |
| `verify` | FunctionDef | См. реализацию | [L262](../../../../scripts/chromaseed_inner_passes.py#L262) |
| `main` | FunctionDef | См. реализацию | [L303](../../../../scripts/chromaseed_inner_passes.py#L303) |
