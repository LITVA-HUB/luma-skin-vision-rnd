# `scripts/chromaseed_gate_stability.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_gate_stability.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed-model sensitivity under bounded encoded-RGB affine contractions.

SHA-256 исходника: `78aced77a445b728cf9d8d6da7095936992bec363584f32a58980778a2a15d88`. Строк: **203**.

## Зависимости

```python
from __future__ import annotations
import itertools
import numpy as np
from chromaseed_gated import predict
from chromaseed_kernel import coordinates, gaussian_kernel
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/chromaseed_gate_stability.py#L13)

```python
ANCHORS = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
```

[Строка 14](../../../../scripts/chromaseed_gate_stability.py#L14)

```python
DOSES = (1 / 255, 4 / 255, 16 / 255, 64 / 255)
```

[Строка 15](../../../../scripts/chromaseed_gate_stability.py#L15)

```python
EPS = 1e-4
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `feature_direction` | FunctionDef | См. реализацию | [L18](../../../../scripts/chromaseed_gate_stability.py#L18) |
| `basic_legal` | FunctionDef | См. реализацию | [L27](../../../../scripts/chromaseed_gate_stability.py#L27) |
| `affine_features` | FunctionDef | См. реализацию | [L43](../../../../scripts/chromaseed_gate_stability.py#L43) |
| `grid` | FunctionDef | См. реализацию | [L61](../../../../scripts/chromaseed_gate_stability.py#L61) |
| `gate_score` | FunctionDef | См. реализацию | [L69](../../../../scripts/chromaseed_gate_stability.py#L69) |
| `projection` | FunctionDef | См. реализацию | [L76](../../../../scripts/chromaseed_gate_stability.py#L76) |
| `describe` | FunctionDef | См. реализацию | [L87](../../../../scripts/chromaseed_gate_stability.py#L87) |
| `summaries` | FunctionDef | См. реализацию | [L113](../../../../scripts/chromaseed_gate_stability.py#L113) |
| `boundaries` | FunctionDef | См. реализацию | [L143](../../../../scripts/chromaseed_gate_stability.py#L143) |
