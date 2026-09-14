# `scripts/chromaseed_refine_numpy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_refine_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent NumPy batch-one deployment path, including actual early stopping.

SHA-256 исходника: `320fcab7cff60e3ef3bc8758ee1777e6d33455a18f76d8e66f64caf35e884e28`. Строк: **104**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `NumpyRefiner` | — | [L12](../../../../scripts/chromaseed_refine_numpy.py#L12) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `silu` | FunctionDef | См. реализацию | [L8](../../../../scripts/chromaseed_refine_numpy.py#L8) |
| `NumpyRefiner` | ClassDef | См. реализацию | [L12](../../../../scripts/chromaseed_refine_numpy.py#L12) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L13–45</summary>

```python
def __init__(self, payload):
        self.family = str(payload["family"])
        self.recurrent = self.family.startswith("recur_")
        self.prep = {k: np.asarray(payload[k], np.float32) for k in ("x_mean", "x_std", "t_mean", "t_std", "y_mean", "y_std", "anchor")}
        theta = np.asarray(payload["theta"], np.float32)
        self.threshold = float(payload.get("exit_threshold", 0.))
        if theta.ndim != 1:
            raise ValueError("deployment payload must contain exactly one model")
        specs = []
        if self.family != "stats_mlp":
            specs += [("token1", 18, 32), ("token2", 32, 24)]
        if self.recurrent:
            specs += [("context", 60, 48), ("query", 51, 24), ("key", 24, 24),
                      ("update1", 123, 48), ("update2", 48, 48), ("head", 48, 3)]
        elif self.family == "patch_mlp":
            specs += [("mlp1", 60, 96), ("mlp2", 96, 64), ("mlp3", 64, 48), ("head", 48, 3)]
        elif self.family == "stats_mlp":
            specs += [("mlp1", 36, 96), ("mlp2", 96, 96), ("mlp3", 96, 48), ("head", 48, 3)]
        else:
            raise ValueError("unknown family")
        self.layers = {}
        offset = 0
        for name, incoming, outgoing in specs:
            size = incoming * outgoing
            weight = theta[offset:offset + size].reshape(incoming, outgoing)
            offset += size
            bias = None
            if name != "key":
                bias = theta[offset:offset + outgoing]
                offset += outgoing
            self.layers[name] = weight, bias
        if offset != len(theta):
            raise ValueError("payload length does not match architecture")
```

</details>

<details><summary>predict · L102–104</summary>

```python
def predict(self, color, patches, threshold=None, sparse=True):
        outputs, counts = self.predict_trace(color, patches, threshold, sparse)
        return outputs[-1], len(outputs), counts
```

</details>
