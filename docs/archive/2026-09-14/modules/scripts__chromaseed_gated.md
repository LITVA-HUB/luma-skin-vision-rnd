# `scripts/chromaseed_gated.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gated.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Shared exact Nyström basis with a small input-conditioned residual readout.

SHA-256 исходника: `1034ce1724d545a225b01787b65d5074cc50cda9967317a6db0ab11bcdba942f`. Строк: **266**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_camera_support import row_weights, standardize
from chromaseed_kernel import coordinates, gaussian_kernel, predict_kernel
from chromaseed_perceptual import coupled_ridge, from_theta, geometry, make_basis, prepare
from scipy.linalg import cho_factor, cho_solve
from skin_local_search_train import weights_for
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/chromaseed_gated.py#L14)

```python
BASES = ("norm", "perceptual")
```

[Строка 15](../../../../scripts/chromaseed_gated.py#L15)

```python
ROUTES = ("base", "uniform", "soft", "hard")
```

[Строка 16](../../../../scripts/chromaseed_gated.py#L16)

```python
FAMILIES = tuple(f"{base}_{route}" for base in BASES for route in ROUTES)
```

[Строка 17](../../../../scripts/chromaseed_gated.py#L17)

```python
SEEDS = (17, 29, 43)
```

[Строка 18](../../../../scripts/chromaseed_gated.py#L18)

```python
LAMBDAS = (0.1, 1.0, 10.0)
```

[Строка 19](../../../../scripts/chromaseed_gated.py#L19)

```python
RHOS = (0.25, 0.5, 1.0)
```

[Строка 20](../../../../scripts/chromaseed_gated.py#L20)

```python
RANK = 128
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `candidates` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_gated.py#L23) |
| `model_id` | FunctionDef | См. реализацию | [L34](../../../../scripts/chromaseed_gated.py#L34) |
| `gate_value` | FunctionDef | См. реализацию | [L41](../../../../scripts/chromaseed_gated.py#L41) |
| `fit_gate` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_gated.py#L50) |
| `residual_solutions` | FunctionDef | См. реализацию | [L63](../../../../scripts/chromaseed_gated.py#L63) |
| `base_model` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_gated.py#L116) |
| `package_model` | FunctionDef | См. реализацию | [L130](../../../../scripts/chromaseed_gated.py#L130) |
| `predict` | FunctionDef | См. реализацию | [L151](../../../../scripts/chromaseed_gated.py#L151) |
| `correction_bank` | FunctionDef | См. реализацию | [L163](../../../../scripts/chromaseed_gated.py#L163) |
| `fit_single` | FunctionDef | См. реализацию | [L176](../../../../scripts/chromaseed_gated.py#L176) |
| `fit_bank` | FunctionDef | См. реализацию | [L209](../../../../scripts/chromaseed_gated.py#L209) |
| `flatten` | FunctionDef | См. реализацию | [L256](../../../../scripts/chromaseed_gated.py#L256) |
| `unpack` | FunctionDef | См. реализацию | [L264](../../../../scripts/chromaseed_gated.py#L264) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L151–160</summary>

```python
def predict(model, x):
    if "correction" not in model:
        return predict_kernel(model, x)
    z = coordinates(model, x)
    k = gaussian_kernel(z, model["centers"], float(model["width"]))
    value = k @ model["coefficient"].astype(np.float64)
    route = "soft" if int(model["gate_mode"]) == 1 else "hard"
    gate = gate_value(z, model["gate_beta"], route)
    value += float(model["rho"]) * gate[:, None] * (k @ model["correction"].astype(np.float64))
    return value * model["y_std"] + model["y_mean"]
```

</details>
