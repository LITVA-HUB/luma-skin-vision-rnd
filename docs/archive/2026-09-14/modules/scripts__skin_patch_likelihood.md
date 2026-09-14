# `scripts/skin_patch_likelihood.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_patch_likelihood.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Latent conditional patch distributions without camera or capture-mode input.

SHA-256 исходника: `64a534647f6c76df14be4b94ae0f2f9bb81541ccdeca0fe901a7dc653f310ea7`. Строк: **67**.

## Зависимости

```python
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from skin_appearance_inverse import design
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `bag_loglik` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_patch_likelihood.py#L8) |
| `component_update` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_patch_likelihood.py#L19) |
| `fit_patch` | FunctionDef | См. реализацию | [L32](../../../../scripts/skin_patch_likelihood.py#L32) |
| `likelihood` | FunctionDef | См. реализацию | [L57](../../../../scripts/skin_patch_likelihood.py#L57) |
| `color_output` | FunctionDef | См. реализацию | [L65](../../../../scripts/skin_patch_likelihood.py#L65) |
