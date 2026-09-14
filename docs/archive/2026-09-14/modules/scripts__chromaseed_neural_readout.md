# `scripts/chromaseed_neural_readout.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_neural_readout.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Analytical output fits folded into unchanged small TG networks.

SHA-256 исходника: `b472aa9c4ff225f967417c4ec645befd02b5ae05545be373da4a7e9ba130e639`. Строк: **201**.

## Зависимости

```python
from __future__ import annotations
import time
import numpy as np
from chromaseed_gaussian import (
    export,
    initialize,
    predict,
    prepare,
    train_block,
)
from chromaseed_gaussian import (
    fit_single as tg_fit,
)
from chromaseed_gaussian_numpy import Predictor
from chromaseed_perceptual import local_tensor
from scipy.linalg import cho_factor, cho_solve
from skin_local_search_train import weights_for
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 23](../../../../scripts/chromaseed_neural_readout.py#L23)

```python
SEEDS = (17, 29, 43)
```

[Строка 24](../../../../scripts/chromaseed_neural_readout.py#L24)

```python
EPOCHS = (1, 4, 16)
```

[Строка 25](../../../../scripts/chromaseed_neural_readout.py#L25)

```python
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)
```

[Строка 34](../../../../scripts/chromaseed_neural_readout.py#L34)

```python
FAMILIES = ("norm", "perceptual")
```

[Строка 35](../../../../scripts/chromaseed_neural_readout.py#L35)

```python
ALPHAS = (0.1, 1.0, 10.0)
```

[Строка 36](../../../../scripts/chromaseed_neural_readout.py#L36)

```python
PARAMETERS = {"adam": 0.001, "tagi_full3": 1.0}
```

[Строка 37](../../../../scripts/chromaseed_neural_readout.py#L37)

```python
GROUPS = ("raw36", "mean3")
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `basis_spec` | FunctionDef | См. реализацию | [L40](../../../../scripts/chromaseed_neural_readout.py#L40) |
| `solve` | FunctionDef | См. реализацию | [L49](../../../../scripts/chromaseed_neural_readout.py#L49) |
| `fit_head` | FunctionDef | См. реализацию | [L95](../../../../scripts/chromaseed_neural_readout.py#L95) |
| `representation_bank` | FunctionDef | См. реализацию | [L154](../../../../scripts/chromaseed_neural_readout.py#L154) |
| `fit_single` | FunctionDef | См. реализацию | [L167](../../../../scripts/chromaseed_neural_readout.py#L167) |
| `name_for` | FunctionDef | См. реализацию | [L188](../../../../scripts/chromaseed_neural_readout.py#L188) |
| `choose` | FunctionDef | См. реализацию | [L193](../../../../scripts/chromaseed_neural_readout.py#L193) |
| `choose_policy` | FunctionDef | См. реализацию | [L197](../../../../scripts/chromaseed_neural_readout.py#L197) |
