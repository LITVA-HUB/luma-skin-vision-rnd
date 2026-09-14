# `scripts/chromaseed_local_denoise_numpy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_local_denoise_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Actual target-free ND consumer: one finite color36 vector -> native Lab3.

SHA-256 исходника: `4d519df10b32397a4b473afe2906252fd6fb81823d6d1eb3889737745c73ccdf`. Строк: **91**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Predictor` | — | [L8](../../../../scripts/chromaseed_local_denoise_numpy.py#L8) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Predictor` | ClassDef | См. реализацию | [L8](../../../../scripts/chromaseed_local_denoise_numpy.py#L8) |
| `predict` | FunctionDef | См. реализацию | [L87](../../../../scripts/chromaseed_local_denoise_numpy.py#L87) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L9–52</summary>

```python
def __init__(self, payload):
        sizes = {
            "plain": (1, 36, 69),
            "local2": (2, 39, 32),
            "local4": (4, 39, 16),
            "blind4": (4, 39, 16),
            "e2e4": (4, 39, 16),
        }
        self.family = str(payload["family"])
        if self.family not in sizes:
            raise ValueError("unknown ND family")
        self.k, d, h = sizes[self.family]
        theta = np.asarray(payload["theta"], np.float64)
        if theta.shape != (self.k, (d + 1) * h + (h + 1) * 3) or not np.isfinite(theta).all():
            raise ValueError("invalid ND parameters")
        self.prep = {}
        for key, size in (("x_mean", 36), ("x_std", 36), ("y_mean", 3), ("y_std", 3)):
            value = np.asarray(payload[key], np.float32)
            if (
                value.shape != (size,)
                or not np.isfinite(value).all()
                or (key.endswith("std") and np.any(value <= 0))
            ):
                raise ValueError("invalid normalizer")
            self.prep[key] = value.copy()
        self.layers = []
        for row in theta:
            self.layers.append(
                (
                    row[: d * h].reshape(d, h).copy(),
                    row[d * h : (d + 1) * h].copy(),
                    row[(d + 1) * h : -3].reshape(h, 3).copy(),
                    row[-3:].copy(),
                )
            )
        angle = np.arange(self.k + 1) * np.pi / (2 * self.k)
        self.a, self.s = np.sin(angle), np.cos(angle)
        self.a[0], self.s[0], self.a[-1], self.s[-1] = 0.0, 1.0, 1.0, 0.0
        self.cached_array_bytes = (
            sum(v.nbytes for v in self.prep.values())
            + sum(v.nbytes for layer in self.layers for v in layer)
            + self.a.nbytes
            + self.s.nbytes
        )
```

</details>

<details><summary>__call__ · L80–84</summary>

```python
def __call__(self, x):
        x = np.asarray(x, np.float32)
        if x.shape != (36,) or not np.isfinite(x).all():
            raise ValueError("one finite (36,) vector required")
        return self._run(x, False)
```

</details>

<details><summary>predict · L87–91</summary>

```python
def predict(payload, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty N x 36 required")
    return Predictor(payload)._run(x, True)
```

</details>
