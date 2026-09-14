# `scripts/chromaseed_recurrent_stream_check.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_recurrent_stream_check.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Qualify streaming execution with sealed AS exports and generated features only.

Does not read image rows, target labels, evaluated predictions or live HR models.
No timing, training, threshold calibration or model selection is performed.

SHA-256 исходника: `173eddb44d7d9687a74359a33721bdd39df500cc77fe8aec698327b4cf7d2e62`. Строк: **229**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from chromaseed_recurrent_stream import VARIANTS, StreamingPredictor
from threadpoolctl import threadpool_limits
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 22](../../../../scripts/chromaseed_recurrent_stream_check.py#L22)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 23](../../../../scripts/chromaseed_recurrent_stream_check.py#L23)

```python
OUT = Path("D:/Luma-RnD/chromaseed_recurrent_stream_v1")
```

[Строка 24](../../../../scripts/chromaseed_recurrent_stream_check.py#L24)

```python
AS_SEAL = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1/verification.json"
```

[Строка 25](../../../../scripts/chromaseed_recurrent_stream_check.py#L25)

```python
AS_SHA = "ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9"
```

[Строка 26](../../../../scripts/chromaseed_recurrent_stream_check.py#L26)

```python
AS_RESULTS = ROOT / "experiments/runs/chromaseed_architecture_scale_v1/results.json"
```

[Строка 27](../../../../scripts/chromaseed_recurrent_stream_check.py#L27)

```python
P3 = Path("D:/Luma-RnD/chromaseed_palette_transfer_v1/registration.json")
```

[Строка 28](../../../../scripts/chromaseed_recurrent_stream_check.py#L28)

```python
P3_SHA = "90a9a78bc46f5ca1f40befe0cd04e8861374868d41710c099c7573da7a2add17"
```

[Строка 29](../../../../scripts/chromaseed_recurrent_stream_check.py#L29)

```python
COMPUTE = Path("D:/Luma-RnD/chromaseed_compute_structure_v1/analysis.json")
```

[Строка 30](../../../../scripts/chromaseed_recurrent_stream_check.py#L30)

```python
COMPUTE_SHA = "107d6d7b926e1d6fe1c5478ef79ee6db2339459ef86349316b58f5acaa532b68"
```

[Строка 31](../../../../scripts/chromaseed_recurrent_stream_check.py#L31)

```python
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
```

[Строка 32](../../../../scripts/chromaseed_recurrent_stream_check.py#L32)

```python
SEEDS = (17, 29, 43)
```

[Строка 33](../../../../scripts/chromaseed_recurrent_stream_check.py#L33)

```python
FIXED_LAYERS = ["token1", "token2", "context", "key"]
```

[Строка 34](../../../../scripts/chromaseed_recurrent_stream_check.py#L34)

```python
PASS_LAYERS = ["query", "update1", "update2", "head"]
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `ObservedPredictor` | StreamingPredictor | [L63](../../../../scripts/chromaseed_recurrent_stream_check.py#L63) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L37](../../../../scripts/chromaseed_recurrent_stream_check.py#L37) |
| `binding` | FunctionDef | См. реализацию | [L42](../../../../scripts/chromaseed_recurrent_stream_check.py#L42) |
| `check_bindings` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_recurrent_stream_check.py#L46) |
| `read_json` | FunctionDef | См. реализацию | [L52](../../../../scripts/chromaseed_recurrent_stream_check.py#L52) |
| `array_digest` | FunctionDef | См. реализацию | [L56](../../../../scripts/chromaseed_recurrent_stream_check.py#L56) |
| `ObservedPredictor` | ClassDef | См. реализацию | [L63](../../../../scripts/chromaseed_recurrent_stream_check.py#L63) |
| `work` | FunctionDef | См. реализацию | [L75](../../../../scripts/chromaseed_recurrent_stream_check.py#L75) |
| `qualify_case` | FunctionDef | См. реализацию | [L81](../../../../scripts/chromaseed_recurrent_stream_check.py#L81) |
| `prepare` | FunctionDef | См. реализацию | [L127](../../../../scripts/chromaseed_recurrent_stream_check.py#L127) |
| `main` | FunctionDef | См. реализацию | [L201](../../../../scripts/chromaseed_recurrent_stream_check.py#L201) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L64–66</summary>

```python
def __init__(self, model):
        super().__init__(model)
        self.calls = []
```

</details>
