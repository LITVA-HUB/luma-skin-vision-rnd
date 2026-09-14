# `scripts/chromaseed_projection_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent weighted-design SVD projection, dense kernel and QR/SVD refit.

SHA-256 исходника: `c1d20192e147c9270fc52701527a675724d5001eb469254b4028b725831d749c`. Строк: **134**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from chromaseed_affine_reference import qr_ridge
from chromaseed_gated_audit import reference_gate
from chromaseed_kernel import select_landmarks
from chromaseed_kernel_audit import direct_kernel, norm
from chromaseed_perceptual_audit import balanced
from chromaseed_perceptual_reference import analytic_tensor
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `spectrum` | FunctionDef | См. реализацию | [L14](../../../../scripts/chromaseed_projection_reference.py#L14) |
| `projector` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_projection_reference.py#L28) |
| `latent` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_projection_reference.py#L38) |
| `direct_width` | FunctionDef | См. реализацию | [L50](../../../../scripts/chromaseed_projection_reference.py#L50) |
| `predict` | FunctionDef | См. реализацию | [L59](../../../../scripts/chromaseed_projection_reference.py#L59) |
| `refit` | FunctionDef | См. реализацию | [L77](../../../../scripts/chromaseed_projection_reference.py#L77) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L59–74</summary>

```python
def predict(model, x):
    if "constant_lab" in model:
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    k = direct_kernel(latent(model, x), model["centers"], float(model["width"]))
    value = k @ model["coefficient"].astype(np.float64)
    if "correction" in model:
        raw = norm(model, x)
        signal = np.sum(raw * model["gate_beta"][None, 1:].astype(np.float64), axis=1) + float(
            model["gate_beta"][0]
        )
        value += (
            float(model["rho"])
            * np.clip(signal, -1, 1)[:, None]
            * (k @ model["correction"].astype(np.float64))
        )
    return value * model["y_std"] + model["y_mean"]
```

</details>
