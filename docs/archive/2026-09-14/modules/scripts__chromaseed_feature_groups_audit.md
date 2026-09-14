# `scripts/chromaseed_feature_groups_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_feature_groups_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent FG payload, geometry, person selection, consumer and QR audit.

SHA-256 исходника: `2044447fb3a341ad58dc55ecb50cbebdcfe64f84680409c5ed999080de07f298`. Строк: **598**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import itertools
import time
from pathlib import Path
import numpy as np
from chromaseed_affine_audit import exact, summaries, validate_normalizer
from chromaseed_feature_groups_numpy import Predictor
from chromaseed_gate_stability_audit import close, transformed
from chromaseed_gated_audit import model_from, reference_gate
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, js, norm, nz
from chromaseed_perceptual_audit import balanced
from chromaseed_projection_reference import direct_width, latent, refit
from chromaseed_projection_reference import predict as reference_predict
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 26](../../../../scripts/chromaseed_feature_groups_audit.py#L26)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 27](../../../../scripts/chromaseed_feature_groups_audit.py#L27)

```python
SOURCE = "f6a8ac340650ae5d62004f59916ba720b702e6dc38f1fedf6928f5797b53f4cf"
```

[Строка 28](../../../../scripts/chromaseed_feature_groups_audit.py#L28)

```python
GROUPS = {
    "raw36": list(range(36)),
    "mean3": [27, 28, 29],
    "median3": [12, 13, 14],
    "central9": list(range(9, 18)),
    "mean_std6": list(range(27, 33)),
    "quant27": list(range(27)),
    "no_corr33": list(range(33)),
}
```

[Строка 37](../../../../scripts/chromaseed_feature_groups_audit.py#L37)

```python
ALL_GROUPS = (*GROUPS, "projected16")
```

[Строка 38](../../../../scripts/chromaseed_feature_groups_audit.py#L38)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```

[Строка 40](../../../../scripts/chromaseed_feature_groups_audit.py#L40)

```python
SETTINGS = [(0.0, np.zeros(3))] + [
    (v / 255, np.array(a)) for v in (1, 4, 16, 64) for a in itertools.product((0.0, 1.0), repeat=3)
]
```

[Строка 43](../../../../scripts/chromaseed_feature_groups_audit.py#L43)

```python
ERRATUM = "docs/research/chromaseed_feature_groups_count_erratum.md"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ident` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_feature_groups_audit.py#L46) |
| `row_hash` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_feature_groups_audit.py#L50) |
| `subset` | FunctionDef | См. реализацию | [L54](../../../../scripts/chromaseed_feature_groups_audit.py#L54) |
| `direct` | FunctionDef | См. реализацию | [L58](../../../../scripts/chromaseed_feature_groups_audit.py#L58) |
| `gate` | FunctionDef | См. реализацию | [L64](../../../../scripts/chromaseed_feature_groups_audit.py#L64) |
| `cached_output` | FunctionDef | См. реализацию | [L72](../../../../scripts/chromaseed_feature_groups_audit.py#L72) |
| `scoring` | FunctionDef | См. реализацию | [L85](../../../../scripts/chromaseed_feature_groups_audit.py#L85) |
| `compare` | FunctionDef | См. реализацию | [L93](../../../../scripts/chromaseed_feature_groups_audit.py#L93) |
| `main` | FunctionDef | См. реализацию | [L108](../../../../scripts/chromaseed_feature_groups_audit.py#L108) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>gate · L64–69</summary>

```python
def gate(model, x):
    xx = x if "feature_indices" not in model else x[:, model["feature_indices"]]
    z = norm(model, xx)
    return np.sum(z * model["gate_beta"][None, 1:].astype(np.float64), axis=1) + float(
        model["gate_beta"][0]
    )
```

</details>
