# `scripts/chromaseed_gaussian.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Small independent FP64 Adam/Gaussian banks; deployment keeps FP32 means only.

SHA-256 исходника: `5359f71766201951c72d4ecbe0447ff869d9bc4841d013106523c3938915695d`. Строк: **278**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import time
import numpy as np
from skin_local_search_train import weights_for
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/chromaseed_gaussian.py#L11)

```python
KEYS = ("w1", "b1", "w2", "b2")
```

[Строка 12](../../../../scripts/chromaseed_gaussian.py#L12)

```python
GROUPS = {"raw36": list(range(36)), "mean3": [27, 28, 29]}
```

[Строка 13](../../../../scripts/chromaseed_gaussian.py#L13)

```python
METHODS = ("adam", "tagi_diag", "tagi_full3")
```

[Строка 14](../../../../scripts/chromaseed_gaussian.py#L14)

```python
SEEDS = (17, 29, 43)
```

[Строка 15](../../../../scripts/chromaseed_gaussian.py#L15)

```python
CHECKPOINTS = (1, 4, 16, 64)
```

[Строка 16](../../../../scripts/chromaseed_gaussian.py#L16)

```python
PARAMETERS = {
    "adam": (0.0003, 0.001, 0.003),
    "tagi_diag": (0.1, 0.3, 1.0),
    "tagi_full3": (0.1, 0.3, 1.0),
}
```

[Строка 21](../../../../scripts/chromaseed_gaussian.py#L21)

```python
VAR_FLOOR = 1e-12
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `initialize` | FunctionDef | См. реализацию | [L24](../../../../scripts/chromaseed_gaussian.py#L24) |
| `activations` | FunctionDef | См. реализацию | [L38](../../../../scripts/chromaseed_gaussian.py#L38) |
| `gaussian_step` | FunctionDef | См. реализацию | [L46](../../../../scripts/chromaseed_gaussian.py#L46) |
| `adam_step` | FunctionDef | См. реализацию | [L93](../../../../scripts/chromaseed_gaussian.py#L93) |
| `prepare` | FunctionDef | См. реализацию | [L116](../../../../scripts/chromaseed_gaussian.py#L116) |
| `export` | FunctionDef | См. реализацию | [L143](../../../../scripts/chromaseed_gaussian.py#L143) |
| `predict` | FunctionDef | См. реализацию | [L150](../../../../scripts/chromaseed_gaussian.py#L150) |
| `train_block` | FunctionDef | См. реализацию | [L162](../../../../scripts/chromaseed_gaussian.py#L162) |
| `fit_single` | FunctionDef | См. реализацию | [L261](../../../../scripts/chromaseed_gaussian.py#L261) |
| `model_id` | FunctionDef | См. реализацию | [L269](../../../../scripts/chromaseed_gaussian.py#L269) |
| `choose` | FunctionDef | См. реализацию | [L273](../../../../scripts/chromaseed_gaussian.py#L273) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>export · L143–147</summary>

```python
def export(prep, mean):
    return {
        **{k: v.copy() for k, v in prep.items()},
        **{k: mean[k].astype(np.float32) for k in KEYS},
    }
```

</details>

<details><summary>predict · L150–159</summary>

```python
def predict(model, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not np.isfinite(x).all():
        raise ValueError("finite N by36 input required")
    if "feature_indices" in model:
        x = x[:, model["feature_indices"]]
    z = ((x - model["x_mean"]) / model["x_std"]).astype(np.float64)
    hidden = np.maximum(z @ model["w1"].astype(np.float64).T + model["b1"].astype(np.float64), 0)
    value = hidden @ model["w2"].astype(np.float64).T + model["b2"].astype(np.float64)
    return value * model["y_std"] + model["y_mean"]
```

</details>
