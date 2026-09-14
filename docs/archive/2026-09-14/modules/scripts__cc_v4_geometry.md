# `scripts/cc_v4_geometry.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_v4_geometry.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Grounded correction-action targets, not a learned model or novel theorem.

An action is log(R/G),log(B/G) of a candidate illuminant; it implies a
positive diagonal correction. Targets follow a supplied real illuminant GT.
They measure neutral reproduction, never physical surface DeltaE. Multiple
queries reuse one GT and do not create independent labeled scenes.

SHA-256 исходника: `5f0aceda35209b40d4f93be6825c329b0c274f83671aaaeb945ab6d1bc95408f`. Строк: **48**.

## Зависимости

```python
import numpy as np
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `correction_targets` | FunctionDef | Return per-action angle, smooth sin²(angle) and analytic action gradient.  gt: Nx3 positive finite camera RGB. actions: NxKx2 finite log ratios, bounded to absolute value30 to keep the declared numerical domain finite. A joint diagonal gain shifts GT and action log ratios by the same amount. That identity is not a claim for arbitrary unknown nonlinear image ISPs. | [L12](../../../../scripts/cc_v4_geometry.py#L12) |
