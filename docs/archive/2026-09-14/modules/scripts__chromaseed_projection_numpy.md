# `scripts/chromaseed_projection_numpy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_projection_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

NumPy-only color36 consumer with optional compact latent kernel coordinates.

SHA-256 исходника: `2669e378ea2882093b41e8aaefb4f0571e95e017b11dd91311b4d24389d2de02`. Строк: **93**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Predictor` | — | [L8](../../../../scripts/chromaseed_projection_numpy.py#L8) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Predictor` | ClassDef | См. реализацию | [L8](../../../../scripts/chromaseed_projection_numpy.py#L8) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L9–67</summary>

```python
def __init__(self, model):
        self.constant = None
        if set(model) == {"constant_lab"}:
            v = model["constant_lab"]
            if v.shape != (3,) or v.dtype != np.float32 or not np.isfinite(v).all():
                raise ValueError("finite FP32 Lab constant required")
            self.constant = v.astype(np.float64)
            return
        base = {"x_mean", "x_std", "y_mean", "y_std", "centers", "coefficient", "width"}
        projection = {"projection", "projection_mean"}
        extra = {"correction", "gate_beta", "gate_mode", "rho"}
        if set(model) not in (base, base | projection, base | extra, base | projection | extra):
            raise ValueError("incomplete or unknown fields")
        if any(
            np.asarray(v).dtype != (np.uint8 if k == "gate_mode" else np.float32)
            or not np.isfinite(v).all()
            for k, v in model.items()
        ):
            raise ValueError("finite FP32 arrays and uint8 mode required")
        if model["centers"].ndim != 2:
            raise ValueError("matrix centers required")
        n, d = model["centers"].shape
        shapes = dict(
            x_mean=(36,),
            x_std=(36,),
            y_mean=(3,),
            y_std=(3,),
            centers=(n, d),
            coefficient=(n, 3),
            width=(),
        )
        if "projection" in model:
            shapes.update(projection=(36, d), projection_mean=(36,))
        elif d != 36:
            raise ValueError("raw coordinates require36 columns")
        if "correction" in model:
            shapes.update(correction=(n, 3), gate_beta=(37,), gate_mode=(), rho=())
        if n < 1 or not 1 <= d <= 36 or any(model[k].shape != shape for k, shape in shapes.items()):
            raise ValueError("incompatible shapes")
        if float(model["width"]) <= 0 or np.any(model["x_std"] <= 0) or np.any(model["y_std"] <= 0):
            raise ValueError("positive scales required")
        self.x_mean, self.x_std, self.y_mean, self.y_std = (
            model[k].copy() for k in ("x_mean", "x_std", "y_mean", "y_std")
        )
        self.centers = model["centers"].astype(np.float64)
        self.norms = np.sum(self.centers * self.centers, axis=1)
        self.coefficient = model["coefficient"].astype(np.float64)
        self.denominator = d * float(model["width"]) ** 2
        self.projection = None
        if "projection" in model:
            self.projection = model["projection"].astype(np.float64)
            self.projection_mean = model["projection_mean"].astype(np.float64)
        self.correction = None
        if "correction" in model:
            if int(model["gate_mode"]) != 1 or not 0 <= float(model["rho"]) <= 1:
                raise ValueError("continuous gate with valid strength required")
            self.correction = model["correction"].astype(np.float64)
            self.beta = model["gate_beta"].astype(np.float64)
            self.rho = float(model["rho"])
```

</details>

<details><summary>__call__ · L73–93</summary>

```python
def __call__(self, color):
        color = np.asarray(color, np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        if self.constant is not None:
            return self.constant.copy()
        z = ((color - self.x_mean) / self.x_std).astype(np.float64)
        q = (
            z
            if self.projection is None
            else ((z - self.projection_mean) @ self.projection)
            .astype(np.float32)
            .astype(np.float64)
        )
        distance = np.maximum(np.sum(q * q) + self.norms - 2 * (self.centers @ q), 0)
        k = np.exp(-0.5 * distance / self.denominator)
        value = k @ self.coefficient
        if self.correction is not None:
            signal = min(max(float(z @ self.beta[1:] + self.beta[0]), -1), 1)
            value += self.rho * signal * (k @ self.correction)
        return value * self.y_std + self.y_mean
```

</details>
