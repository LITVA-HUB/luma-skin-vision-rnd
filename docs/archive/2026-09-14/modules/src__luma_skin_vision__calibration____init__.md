# `src/luma_skin_vision/calibration/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/calibration/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Split calibration on subject maxima; no conditional selective-risk guarantee.

Finite-sample one-sided bound assumes exchangeable subject bundles and frozen
predictor, capture protocol and score. It does not guarantee safety under shift.

SHA-256 исходника: `d88bcd2418b230bc1c206142fbe8985a8c31e51b2e35d286c4ee8140c7e022ed`. Строк: **88**.

## Зависимости

```python
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Calibrator` | — | [L16](../../../../src/luma_skin_vision/calibration/__init__.py#L16) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Calibrator` | ClassDef | См. реализацию | [L16](../../../../src/luma_skin_vision/calibration/__init__.py#L16) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L46–70</summary>

```python
def fit(
        cls,
        predicted_error,
        observed_error,
        subjects,
        *,
        split,
        model_hash,
        data_kind,
        alpha=0.1,
        tolerance=5,
    ):
        if split != "calibration":
            raise ValueError("calibration split required")
        p, y, s = np.asarray(predicted_error), np.asarray(observed_error), np.asarray(subjects)
        if p.ndim != 1 or p.shape != y.shape or len(p) != len(s) or not len(p):
            raise ValueError("invalid calibration vectors")
        if not np.isfinite(p).all() or not np.isfinite(y).all() or np.any(p < 0) or np.any(y < 0):
            raise ValueError("invalid calibration scores")
        if not 0 < alpha < 1 or tolerance <= 0:
            raise ValueError("invalid alpha or tolerance")
        scores = sorted(float((y - p)[s == k].max()) for k in np.unique(s))
        k = math.ceil((len(scores) + 1) * (1 - alpha))
        q = scores[k - 1] if k <= len(scores) else None
        return cls("1.0", model_hash, data_kind, alpha, tolerance, q, len(scores))
```

</details>
