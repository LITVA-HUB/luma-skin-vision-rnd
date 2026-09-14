# `scripts/skin_local_search_core.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_local_search_core.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Float64 closed-form primitives for bounded local skin-model searches.

SHA-256 исходника: `b2f916d67bd6e0a9f0341fa85b39c82457e973edf25cde1398b0281b927cad20`. Строк: **161**.

## Зависимости

```python
import math
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_require_float64` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_local_search_core.py#L8) |
| `_positive_scalar` | FunctionDef | См. реализацию | [L18](../../../../scripts/skin_local_search_core.py#L18) |
| `_validated_weights` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_local_search_core.py#L25) |
| `_regression_inputs` | FunctionDef | См. реализацию | [L36](../../../../scripts/skin_local_search_core.py#L36) |
| `ridge_solve` | FunctionDef | Solve weighted ridge regression, leaving design column zero unpenalized. | [L48](../../../../scripts/skin_local_search_core.py#L48) |
| `regularized_objective` | FunctionDef | Return weighted squared error plus ridge penalty, excluding row zero of beta. | [L58](../../../../scripts/skin_local_search_core.py#L58) |
| `rbf_features` | FunctionDef | Evaluate isotropic Gaussian features using mean squared coordinate distance. | [L69](../../../../scripts/skin_local_search_core.py#L69) |
| `greedy_ridge_indices` | FunctionDef | Select ridge atoms by exact Schur-complement objective reductions. | [L87](../../../../scripts/skin_local_search_core.py#L87) |
| `kernel_ridge_fit` | FunctionDef | Fit dual coefficients for weighted Gaussian kernel ridge without a bias. | [L136](../../../../scripts/skin_local_search_core.py#L136) |
| `kernel_ridge_predict` | FunctionDef | Predict from Gaussian kernel ridge coefficients. | [L151](../../../../scripts/skin_local_search_core.py#L151) |
