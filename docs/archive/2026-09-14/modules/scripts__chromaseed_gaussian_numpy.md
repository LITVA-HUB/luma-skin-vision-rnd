# `scripts/chromaseed_gaussian_numpy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gaussian_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Validated NumPy-only deterministic consumer for the TG mean network.

SHA-256 исходника: `725abb85e11bca359e7323bd8fa491c2e8b8d6c9b020a93a90567601691bd4d5`. Строк: **56**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Predictor` | — | [L8](../../../../scripts/chromaseed_gaussian_numpy.py#L8) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Predictor` | ClassDef | См. реализацию | [L8](../../../../scripts/chromaseed_gaussian_numpy.py#L8) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L9–43</summary>

```python
def __init__(self, model):
        required = {"w1", "b1", "w2", "b2", "x_mean", "x_std", "y_mean", "y_std"}
        if set(model) not in (required, required | {"feature_indices"}):
            raise ValueError("complete mean-network schema required")
        for key, value in model.items():
            dtype = np.uint8 if key == "feature_indices" else np.float32
            if np.asarray(value).dtype != dtype or not np.isfinite(value).all():
                raise ValueError("finite arrays with prescribed storage dtype required")
        if model["w1"].ndim != 2:
            raise ValueError("matrix first-layer weight required")
        hidden, d = model["w1"].shape
        shapes = dict(
            b1=(hidden,), w2=(3, hidden), b2=(3,), x_mean=(d,), x_std=(d,), y_mean=(3,), y_std=(3,)
        )
        if hidden < 1 or any(model[k].shape != v for k, v in shapes.items()):
            raise ValueError("incompatible network/normalizer shapes")
        self.indices = None
        if "feature_indices" in model:
            indices = model["feature_indices"]
            if (
                indices.shape != (d,)
                or not 1 <= d < 36
                or np.any(indices >= 36)
                or np.any(np.diff(indices.astype(int)) <= 0)
            ):
                raise ValueError("increasing distinct input coordinates required")
            self.indices = indices.copy()
        elif d != 36:
            raise ValueError("full input without indices must have dimension36")
        if np.any(model["x_std"] <= 0) or np.any(model["y_std"] <= 0):
            raise ValueError("positive normalization required")
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            setattr(self, key, model[key].copy())
        for key in ("w1", "b1", "w2", "b2"):
            setattr(self, key, model[key].astype(np.float64))
```

</details>

<details><summary>__call__ · L49–56</summary>

```python
def __call__(self, color):
        color = np.asarray(color, np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        x = color if self.indices is None else color[self.indices]
        z = ((x - self.x_mean) / self.x_std).astype(np.float64)
        hidden = np.maximum(self.w1 @ z + self.b1, 0)
        return (self.w2 @ hidden + self.b2) * self.y_std + self.y_mean
```

</details>
