# `scripts/chromaseed_gated_numpy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gated_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Standalone ChromaSeed-G inference. Imports NumPy only; input is color36, not an image.

SHA-256 исходника: `5c2472cc6bbd759bdc81df95d7d25a440c3e52b60e6108377946c26c14c2e4af`. Строк: **70**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Predictor` | — | [L8](../../../../scripts/chromaseed_gated_numpy.py#L8) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Predictor` | ClassDef | См. реализацию | [L8](../../../../scripts/chromaseed_gated_numpy.py#L8) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L9–52</summary>

```python
def __init__(self, model):
        base = {"x_mean", "x_std", "y_mean", "y_std", "centers", "coefficient", "width"}
        extra = {"correction", "gate_beta", "gate_mode", "rho"}
        if set(model) not in (base, base | extra):
            raise ValueError("incomplete or unknown payload fields")
        if not all(np.isfinite(v).all() for v in model.values()):
            raise ValueError("nonfinite model")
        n = len(model["centers"])
        shapes = dict(
            x_mean=(36,),
            x_std=(36,),
            y_mean=(3,),
            y_std=(3,),
            centers=(n, 36),
            coefficient=(n, 3),
            width=(),
        )
        if set(model) == base | extra:
            shapes.update(correction=(n, 3), gate_beta=(37,), gate_mode=(), rho=())
        if n < 1 or any(np.shape(model[k]) != shape for k, shape in shapes.items()):
            raise ValueError("incompatible model shapes")
        if np.any(model["x_std"] <= 0) or np.any(model["y_std"] <= 0) or float(model["width"]) <= 0:
            raise ValueError("positive scales required")
        if any(
            np.asarray(v).dtype != (np.uint8 if k == "gate_mode" else np.float32)
            for k, v in model.items()
        ):
            raise ValueError("expected frozen FP32 arrays and uint8 gate mode")
        self.x_mean = model["x_mean"].copy()
        self.x_std = model["x_std"].copy()
        self.y_mean = model["y_mean"].copy()
        self.y_std = model["y_std"].copy()
        self.centers = model["centers"].astype(np.float64)
        self.center_norm = (self.centers * self.centers).sum(axis=1)
        self.coefficient = model["coefficient"].astype(np.float64)
        self.denominator = 36 * float(model["width"]) ** 2
        self.correction = None
        if "correction" in model:
            if int(model["gate_mode"]) not in (1, 2) or not 0.0 <= float(model["rho"]) <= 1.0:
                raise ValueError("invalid gate mode/strength")
            self.correction = model["correction"].astype(np.float64)
            self.beta = model["gate_beta"].astype(np.float64)
            self.mode = int(model["gate_mode"])
            self.rho = float(model["rho"])
```

</details>

<details><summary>__call__ · L58–70</summary>

```python
def __call__(self, color):
        color = np.asarray(color, dtype=np.float32)
        if color.shape != (36,) or not np.isfinite(color).all():
            raise ValueError("one finite color36 vector required")
        z = ((color - self.x_mean) / self.x_std).astype(np.float64)
        distance = np.maximum(np.sum(z * z) + self.center_norm - 2.0 * (self.centers @ z), 0.0)
        k = np.exp(-0.5 * distance / self.denominator)
        value = k @ self.coefficient
        if self.correction is not None:
            score = float(z @ self.beta[1:] + self.beta[0])
            gate = min(max(score, -1.0), 1.0) if self.mode == 1 else (1.0 if score >= 0.0 else -1.0)
            value += self.rho * gate * (k @ self.correction)
        return value * self.y_std + self.y_mean
```

</details>
