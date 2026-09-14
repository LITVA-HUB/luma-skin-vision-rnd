# `scripts/chromaseed_neural_prefix_numpy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_prefix_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

NP exact ND prefix export and standalone NumPy one-color36 consumer.

SHA-256 исходника: `b6e6751eaf45a39e62a89941d60166989b2ac8dd4b3364470b76ae0c3e18e1d4`. Строк: **150**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 7](../../../../scripts/chromaseed_neural_prefix_numpy.py#L7)

```python
SPECS = {
    "plain": (1, 36, 69),
    "local2": (2, 39, 32),
    "local4": (4, 39, 16),
    "blind4": (4, 39, 16),
    "e2e4": (4, 39, 16),
}
```

[Строка 14](../../../../scripts/chromaseed_neural_prefix_numpy.py#L14)

```python
TOLERANCE = 0.10
```

[Строка 15](../../../../scripts/chromaseed_neural_prefix_numpy.py#L15)

```python
PREP = (("x_mean", 36), ("x_std", 36), ("y_mean", 3), ("y_std", 3))
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Predictor` | — | [L70](../../../../scripts/chromaseed_neural_prefix_numpy.py#L70) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `capacity` | FunctionDef | См. реализацию | [L18](../../../../scripts/chromaseed_neural_prefix_numpy.py#L18) |
| `choose` | FunctionDef | См. реализацию | [L26](../../../../scripts/chromaseed_neural_prefix_numpy.py#L26) |
| `export_prefix` | FunctionDef | См. реализацию | [L39](../../../../scripts/chromaseed_neural_prefix_numpy.py#L39) |
| `Predictor` | ClassDef | Cached FP64 math, FP32 normalization, one finite (36,) input -> Lab3. | [L70](../../../../scripts/chromaseed_neural_prefix_numpy.py#L70) |
| `predict` | FunctionDef | См. реализацию | [L146](../../../../scripts/chromaseed_neural_prefix_numpy.py#L146) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>capacity · L18–23</summary>

```python
def capacity(payload):
    return dict(
        parameters=sum(v.size for k, v in payload.items() if k[0] in "wbvc"),
        numeric_bytes=sum(v.nbytes for v in payload.values() if v.dtype.kind in "biufc"),
        executed_blocks=sum(k.startswith("w") for k in payload),
    )
```

</details>

<details><summary>__init__ · L73–120</summary>

```python
def __init__(self, payload):
        family = str(payload["family"])
        if family not in SPECS or np.asarray(payload["family"]).shape != ():
            raise ValueError("invalid family")
        k, _, h = SPECS[family]
        for key in ("original_k", "prefix"):
            v = np.asarray(payload[key])
            if v.shape != () or v.dtype != np.uint8:
                raise ValueError("uint8 scalar schedule metadata required")
        j = int(payload["prefix"])
        if int(payload["original_k"]) != k or not 1 <= j <= k:
            raise ValueError("invalid original schedule or prefix")
        count = 1 if family == "blind4" else j
        expected = {"family", "original_k", "prefix", *(key for key, _ in PREP)}
        expected.update(f"{key}{i}" for i in range(count) for key in "wbvc")
        if set(payload) != expected:
            raise ValueError("unexpected or missing payload arrays")
        self.prep = {}
        for key, size in PREP:
            v = np.asarray(payload[key])
            if (
                v.shape != (size,)
                or v.dtype != np.float32
                or not np.isfinite(v).all()
                or (key.endswith("std") and np.any(v <= 0))
            ):
                raise ValueError("invalid normalizer")
            self.prep[key] = v.copy()
        self.layers = []
        for i in range(count):
            layer = []
            for key, shape in zip(
                "wbvc", ((36 if i == 0 else 39, h), (h,), (h, 3), (3,)), strict=True
            ):
                v = np.asarray(payload[f"{key}{i}"])
                if v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all():
                    raise ValueError("invalid weights")
                layer.append(v.astype(np.float64))
            self.layers.append(tuple(layer))
        angle = np.arange(count) * np.pi / (2 * k)
        self.ratios = np.cos(angle[1:]) / np.cos(angle[:-1])
        self.scales = np.sin(angle[1:]) - self.ratios * np.sin(angle[:-1])
        self.cached_array_bytes = (
            sum(v.nbytes for v in self.prep.values())
            + sum(v.nbytes for layer in self.layers for v in layer)
            + self.ratios.nbytes
            + self.scales.nbytes
        )
```

</details>

<details><summary>__call__ · L139–143</summary>

```python
def __call__(self, x):
        x = np.asarray(x, np.float32)
        if x.shape != (36,) or not np.isfinite(x).all():
            raise ValueError("one finite (36,) vector required")
        return self._run(x)
```

</details>

<details><summary>predict · L146–150</summary>

```python
def predict(payload, x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty N x 36 required")
    return Predictor(payload)._run(x)
```

</details>
