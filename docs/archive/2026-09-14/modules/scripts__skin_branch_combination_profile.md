# `scripts/skin_branch_combination_profile.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_branch_combination_profile.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Batch-one prepared-token latency of all fixed two-model combinations.

SHA-256 исходника: `6f0d258d02a0901dccfb3e409a1b31bc6730f355db5556b67fa8b22db9e582a0`. Строк: **51**.

## Зависимости

```python
import os
import json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_capture_model import CaptureColor
from skin_train_branch_model import TrainingBranchColor
from skin_branch_combination import PAIRS,OUT
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_branch_combination_profile.py#L15) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L29–32</summary>

```python
def predict():
            a=models[0](x)[0]*scales[0][1]+scales[0][0]
            b=models[1](x)[0]*scales[1][1]+scales[1][0]
            return .5*(a+b)
```

</details>
