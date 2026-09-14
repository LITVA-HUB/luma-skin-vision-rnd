# `scripts/chromaseed_patch8_numpy.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_patch8_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

One-pass P8 consumer: NP color head plus a small local-patch branch.

SHA-256 исходника: `d0a5c0f4a3a40d918f96b3a45e1ad75f5c9dfce86b7ff5656f128e055c2f1243`. Строк: **108**.

## Зависимости

```python
from __future__ import annotations
import numpy as np
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../scripts/chromaseed_patch8_numpy.py#L8)

```python
EXTRA = {"u", "e", "g", "t_mean", "t_std"}
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Predictor` | — | [L31](../../../../scripts/chromaseed_patch8_numpy.py#L31) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `transform_tokens` | FunctionDef | См. реализацию | [L11](../../../../scripts/chromaseed_patch8_numpy.py#L11) |
| `Predictor` | ClassDef | См. реализацию | [L31](../../../../scripts/chromaseed_patch8_numpy.py#L31) |
| `predict` | FunctionDef | См. реализацию | [L92](../../../../scripts/chromaseed_patch8_numpy.py#L92) |
| `choose` | FunctionDef | См. реализацию | [L104](../../../../scripts/chromaseed_patch8_numpy.py#L104) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L32–58</summary>

```python
def __init__(self, model):
        self.patch = "u" in model
        base = {k: v for k, v in model.items() if k not in EXTRA}
        self.base = BasePredictor(base)
        if str(model["family"]) != "blind4" or model["w0"].shape != (36, 16):
            raise ValueError("NP643 head required")
        if set(model) != (set(base) | EXTRA if self.patch else set(base)):
            raise ValueError("incomplete local branch")
        self.cached_array_bytes = self.base.cached_array_bytes
        if self.patch:
            for key, shape in (
                ("u", (18, 8)),
                ("e", (8,)),
                ("g", (24, 16)),
                ("t_mean", (18,)),
                ("t_std", (18,)),
            ):
                v = np.asarray(model[key])
                if (
                    v.shape != shape
                    or v.dtype != np.float32
                    or not np.isfinite(v).all()
                    or (key == "t_std" and np.any(v <= 0))
                ):
                    raise ValueError("invalid local branch array")
                setattr(self, key, v.copy() if key.startswith("t_") else v.astype(np.float64))
                self.cached_array_bytes += getattr(self, key).nbytes
```

</details>

<details><summary>__call__ · L78–89</summary>

```python
def __call__(self, x, tokens=None):
        if not self.patch:
            return self.base(x)
        x, tokens = np.asarray(x, np.float32), np.asarray(tokens, np.float32)
        if (
            x.shape != (36,)
            or tokens.shape != (64, 18)
            or not np.isfinite(x).all()
            or not np.isfinite(tokens).all()
        ):
            raise ValueError("one finite color36 and tokens64x18 required")
        return self._run(x, tokens)
```

</details>

<details><summary>predict · L92–101</summary>

```python
def predict(model, x, tokens=None):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != 36 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty colorNx36 required")
    consumer = Predictor(model)
    if consumer.patch:
        tokens = np.asarray(tokens, np.float32)
        if tokens.shape != (len(x), 64, 18) or not np.isfinite(tokens).all():
            raise ValueError("matching finite tokensNx64x18 required")
    return consumer._run(x, tokens)
```

</details>
