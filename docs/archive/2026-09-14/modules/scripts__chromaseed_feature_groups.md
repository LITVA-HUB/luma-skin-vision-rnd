# `scripts/chromaseed_feature_groups.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_feature_groups.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed input subsets with matched A readouts and exact raw/X references.

SHA-256 исходника: `a0d3a11eba62f1e797171d03fe05a652ea53632ff7d01e8d0899d8711f19b33a`. Строк: **216**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_affine import color_metrics, design_matrix, export_model, gram_stats, solve_stats
from chromaseed_gated import fit_gate
from chromaseed_gated import predict as gated_predict
from chromaseed_kernel import coordinates
from chromaseed_perceptual import make_basis, prepare
from chromaseed_projection import exact_width, projected_basis
from chromaseed_projection import predict as x_predict
from skin_local_search_train import weights_for
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/chromaseed_feature_groups.py#L17)

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

[Строка 26](../../../../scripts/chromaseed_feature_groups.py#L26)

```python
ALL_GROUPS = (*GROUPS, "projected16")
```

[Строка 27](../../../../scripts/chromaseed_feature_groups.py#L27)

```python
FAMILIES = ("norm_static", "norm_joint_soft", "perceptual_static", "perceptual_joint_soft")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `gather` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_feature_groups.py#L31) |
| `state_for` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_feature_groups.py#L38) |
| `basis_for` | FunctionDef | См. реализацию | [L62](../../../../scripts/chromaseed_feature_groups.py#L62) |
| `export` | FunctionDef | См. реализацию | [L73](../../../../scripts/chromaseed_feature_groups.py#L73) |
| `predict` | FunctionDef | См. реализацию | [L80](../../../../scripts/chromaseed_feature_groups.py#L80) |
| `gate_score` | FunctionDef | См. реализацию | [L90](../../../../scripts/chromaseed_feature_groups.py#L90) |
| `model_id` | FunctionDef | См. реализацию | [L101](../../../../scripts/chromaseed_feature_groups.py#L101) |
| `choose_alpha` | FunctionDef | См. реализацию | [L105](../../../../scripts/chromaseed_feature_groups.py#L105) |
| `choose_policy` | FunctionDef | См. реализацию | [L111](../../../../scripts/chromaseed_feature_groups.py#L111) |
| `fit_bank` | FunctionDef | См. реализацию | [L123](../../../../scripts/chromaseed_feature_groups.py#L123) |
| `fit_single` | FunctionDef | См. реализацию | [L195](../../../../scripts/chromaseed_feature_groups.py#L195) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>export · L73–77</summary>

```python
def export(state, basis, theta, beta, joint, group):
    model = export_model(state, basis, theta, beta, joint)
    if group != "raw36":
        model["feature_indices"] = np.asarray(GROUPS[group], np.uint8)
    return model
```

</details>

<details><summary>predict · L80–87</summary>

```python
def predict(model, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
        raise ValueError("finite color36 matrix required")
    if "feature_indices" not in model:
        return x_predict(model, x)
    base = {k: v for k, v in model.items() if k != "feature_indices"}
    return gated_predict(base, x[:, model["feature_indices"]])
```

</details>
