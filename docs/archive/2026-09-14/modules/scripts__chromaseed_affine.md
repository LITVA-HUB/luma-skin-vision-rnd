# `scripts/chromaseed_affine.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_affine.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Compact analytical static/joint readouts with fit-only mild affine augmentation.

SHA-256 исходника: `df0757077715a6764cf4cd066dfd8aa188bad6dab0d7f532b7fd834acc817c0e`. Строк: **264**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_gate_stability import ANCHORS, affine_features
from chromaseed_gated import fit_gate, package_model
from chromaseed_gated import predict as gated_predict
from chromaseed_kernel import coordinates, gaussian_kernel
from chromaseed_perceptual import from_theta, geometry, make_basis, prepare
from skin_local_search_train import weights_for
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/chromaseed_affine.py#L15)

```python
BASES = ("norm", "perceptual")
```

[Строка 16](../../../../scripts/chromaseed_affine.py#L16)

```python
FAMILIES = tuple(f"{b}_{s}" for b in BASES for s in ("static", "joint_soft"))
```

[Строка 17](../../../../scripts/chromaseed_affine.py#L17)

```python
ALPHAS = (0.1, 1.0, 10.0)
```

[Строка 18](../../../../scripts/chromaseed_affine.py#L18)

```python
ETAS = (0.0, 0.25, 0.5, 0.75)
```

[Строка 19](../../../../scripts/chromaseed_affine.py#L19)

```python
SEEDS = (17, 29, 43)
```

[Строка 20](../../../../scripts/chromaseed_affine.py#L20)

```python
TRAIN_DOSES = (1 / 255, 4 / 255)
```

[Строка 21](../../../../scripts/chromaseed_affine.py#L21)

```python
POLICIES = ("clean", "guarded")
```

[Строка 22](../../../../scripts/chromaseed_affine.py#L22)

```python
G_CONTROLS = ("norm_base", "norm_soft", "perceptual_base", "perceptual_soft")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `weighted_copies` | FunctionDef | См. реализацию | [L25](../../../../scripts/chromaseed_affine.py#L25) |
| `gram_stats` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_affine.py#L38) |
| `combine` | FunctionDef | См. реализацию | [L52](../../../../scripts/chromaseed_affine.py#L52) |
| `solve_stats` | FunctionDef | См. реализацию | [L60](../../../../scripts/chromaseed_affine.py#L60) |
| `design_matrix` | FunctionDef | См. реализацию | [L117](../../../../scripts/chromaseed_affine.py#L117) |
| `export_model` | FunctionDef | См. реализацию | [L123](../../../../scripts/chromaseed_affine.py#L123) |
| `predict` | FunctionDef | См. реализацию | [L135](../../../../scripts/chromaseed_affine.py#L135) |
| `model_id` | FunctionDef | См. реализацию | [L144](../../../../scripts/chromaseed_affine.py#L144) |
| `choose` | FunctionDef | См. реализацию | [L148](../../../../scripts/chromaseed_affine.py#L148) |
| `sufficient_stats` | FunctionDef | См. реализацию | [L158](../../../../scripts/chromaseed_affine.py#L158) |
| `color_metrics` | FunctionDef | См. реализацию | [L185](../../../../scripts/chromaseed_affine.py#L185) |
| `fit_bank` | FunctionDef | См. реализацию | [L194](../../../../scripts/chromaseed_affine.py#L194) |
| `fit_single` | FunctionDef | См. реализацию | [L244](../../../../scripts/chromaseed_affine.py#L244) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L135–141</summary>

```python
def predict(model, x):
    if "constant_lab" in model:
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
            raise ValueError("finite color36 batch required")
        return np.broadcast_to(model["constant_lab"].astype(np.float64), (len(x), 3)).copy()
    return gated_predict(model, x)
```

</details>
