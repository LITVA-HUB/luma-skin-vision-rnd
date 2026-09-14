# `scripts/chromaseed_perceptual_reference.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_perceptual_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent analytic infinitesimal CIEDE2000 tensor and augmented LS solve.

Derived from Sharma et al. (2005) equations; no finite-difference implementation
or training normal-equation assembly is called by these reference functions.

SHA-256 исходника: `5b386f30ca16574b64a4cfebd2e940d8f2cbebb2b94ebd1863ff1d96b7f41d0c`. Строк: **43**.

## Зависимости

```python
import numpy as np
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `analytic_tensor` | FunctionDef | См. реализацию | [L9](../../../../scripts/chromaseed_perceptual_reference.py#L9) |
| `augmented_svd` | FunctionDef | См. реализацию | [L35](../../../../scripts/chromaseed_perceptual_reference.py#L35) |
