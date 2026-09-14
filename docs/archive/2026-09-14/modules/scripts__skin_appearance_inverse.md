# `scripts/skin_appearance_inverse.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_appearance_inverse.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Small conditional appearance likelihoods; actual native skin colors as atoms.

SHA-256 исходника: `3ebc84fffdce61e001eaa46f2fddb4a7480b1e8fc9e13aa56f2405f4ee2de172`. Строк: **91**.

## Зависимости

```python
import numpy as np
from scipy.special import logsumexp
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `design` | FunctionDef | См. реализацию | [L7](../../../../scripts/skin_appearance_inverse.py#L7) |
| `fit_map` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_appearance_inverse.py#L14) |
| `predict_map` | FunctionDef | См. реализацию | [L21](../../../../scripts/skin_appearance_inverse.py#L21) |
| `posterior` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_appearance_inverse.py#L25) |
| `decisions` | FunctionDef | См. реализацию | [L42](../../../../scripts/skin_appearance_inverse.py#L42) |
| `image_features` | FunctionDef | См. реализацию | [L50](../../../../scripts/skin_appearance_inverse.py#L50) |
| `fit_forward` | FunctionDef | См. реализацию | [L57](../../../../scripts/skin_appearance_inverse.py#L57) |
| `forward_means` | FunctionDef | См. реализацию | [L81](../../../../scripts/skin_appearance_inverse.py#L81) |
| `infer_forward` | FunctionDef | См. реализацию | [L89](../../../../scripts/skin_appearance_inverse.py#L89) |
