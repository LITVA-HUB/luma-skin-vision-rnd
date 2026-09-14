# `scripts/chromaseed_hybrid.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_hybrid.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Staged analytic corrections on raw/shared-projected landmark geometry.

SHA-256 исходника: `7d078cfa51fb9da79acf61f34a0180d39c0a3e13a78ee81987dbd1c5c65f69c9`. Строк: **287**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_affine import color_metrics, design_matrix, export_model, gram_stats, solve_stats
from chromaseed_gated import fit_gate
from chromaseed_gated import predict as raw_predict
from chromaseed_kernel import coordinates, gaussian_kernel
from chromaseed_perceptual import make_basis, prepare
from chromaseed_projection import exact_width, export, project, projection
from chromaseed_projection import predict as old_predict
from skin_local_search_train import weights_for
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/chromaseed_hybrid.py#L19)

```python
KINDS = ("raw", "projected", "blend", "uniform", "support")
```

[Строка 20](../../../../scripts/chromaseed_hybrid.py#L20)

```python
MODES = {"blend": 1, "uniform": 2, "support": 3}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `support` | FunctionDef | См. реализацию | [L23](../../../../scripts/chromaseed_hybrid.py#L23) |
| `shared_basis` | FunctionDef | См. реализацию | [L32](../../../../scripts/chromaseed_hybrid.py#L32) |
| `model_id` | FunctionDef | См. реализацию | [L53](../../../../scripts/chromaseed_hybrid.py#L53) |
| `settings` | FunctionDef | См. реализацию | [L60](../../../../scripts/chromaseed_hybrid.py#L60) |
| `choose` | FunctionDef | См. реализацию | [L76](../../../../scripts/chromaseed_hybrid.py#L76) |
| `package` | FunctionDef | См. реализацию | [L86](../../../../scripts/chromaseed_hybrid.py#L86) |
| `predict` | FunctionDef | См. реализацию | [L122](../../../../scripts/chromaseed_hybrid.py#L122) |
| `raw_fit` | FunctionDef | См. реализацию | [L153](../../../../scripts/chromaseed_hybrid.py#L153) |
| `projection_fit` | FunctionDef | См. реализацию | [L170](../../../../scripts/chromaseed_hybrid.py#L170) |
| `fit_bank` | FunctionDef | См. реализацию | [L177](../../../../scripts/chromaseed_hybrid.py#L177) |
| `fit_single` | FunctionDef | См. реализацию | [L261](../../../../scripts/chromaseed_hybrid.py#L261) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L122–150</summary>

```python
def predict(model, x):
    if "hybrid_mode" not in model:
        return old_predict(model, x)
    z = coordinates(model, x)
    raw_k = gaussian_kernel(z, model["centers"], float(model["width"]))
    p, mean = model["latent_projection"].astype(np.float64), model["latent_mean"].astype(np.float64)
    q = ((z - mean) @ p).astype(np.float32).astype(np.float64)
    centers = ((model["centers"].astype(np.float64) - mean) @ p).astype(np.float32)
    k = gaussian_kernel(q, centers, float(model["latent_width"]))
    base, correction = (
        raw_k @ model["coefficient"].astype(np.float64),
        k @ model["latent_coefficient"].astype(np.float64),
    )
    if "gate_beta" in model:
        signal = np.clip(
            z @ model["gate_beta"][1:].astype(np.float64) + float(model["gate_beta"][0]), -1, 1
        )[:, None]
        base += signal * (raw_k @ model["correction"].astype(np.float64))
        correction += signal * (k @ model["latent_correction"].astype(np.float64))
    rho = float(model["mix"])
    mode = int(model["hybrid_mode"])
    if mode == 1:
        value = (1 - rho) * base + rho * correction
    elif mode in (2, 3):
        scale = support(raw_k, int(model["support_power"]))[:, None] if mode == 3 else 1.0
        value = base + rho * scale * correction
    else:
        raise ValueError("unsupported hybrid mode")
    return value * model["y_std"] + model["y_mean"]
```

</details>
