# `scripts/chromaseed_gaussian_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent scalar Gaussian smoother / Adam reference; no primary-core import.

SHA-256 исходника: `ecc97d92037587938d3bb5805cf91619958640bd4d26b242dbd5dd854f115fde`. Строк: **178**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import numpy as np
from scipy.linalg import cho_solve
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 10](../../../../scripts/chromaseed_gaussian_reference.py#L10)

```python
KEYS = ("w1", "b1", "w2", "b2")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `initial` | FunctionDef | См. реализацию | [L13](../../../../scripts/chromaseed_gaussian_reference.py#L13) |
| `normalization` | FunctionDef | См. реализацию | [L31](../../../../scripts/chromaseed_gaussian_reference.py#L31) |
| `gaussian_update` | FunctionDef | Condition hidden states first, then incoming parameters by their local gains. | [L51](../../../../scripts/chromaseed_gaussian_reference.py#L51) |
| `adam_update` | FunctionDef | См. реализацию | [L104](../../../../scripts/chromaseed_gaussian_reference.py#L104) |
| `train_reference` | FunctionDef | См. реализацию | [L124](../../../../scripts/chromaseed_gaussian_reference.py#L124) |
| `predict` | FunctionDef | См. реализацию | [L171](../../../../scripts/chromaseed_gaussian_reference.py#L171) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L171–178</summary>

```python
def predict(model, x):
    indices = model.get("feature_indices", np.arange(36))
    z = (np.asarray(x, np.float32)[:, indices] - model["x_mean"]) / model["x_std"]
    hidden = np.maximum(
        0, np.matmul(model["w1"].astype(float), z.astype(float).T) + model["b1"][:, None]
    )
    value = np.matmul(model["w2"].astype(float), hidden) + model["b2"][:, None]
    return value.T * model["y_std"] + model["y_mean"]
```

</details>
