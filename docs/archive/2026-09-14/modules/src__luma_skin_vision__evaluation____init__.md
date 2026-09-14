# `src/luma_skin_vision/evaluation/__init__.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../src/luma_skin_vision/evaluation/__init__.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Evaluation uses real errors, deterministic score ties and subject resampling.

SHA-256 исходника: `21844f2684b4e867b96a3d2677259dc88ef767f011765747c13b924c6f06c8f9`. Строк: **215**.

## Зависимости

```python
import hashlib
import numpy as np
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_vectors` | FunctionDef | См. реализацию | [L10](../../../../src/luma_skin_vision/evaluation/__init__.py#L10) |
| `selection_order` | FunctionDef | См. реализацию | [L24](../../../../src/luma_skin_vision/evaluation/__init__.py#L24) |
| `risk_coverage` | FunctionDef | См. реализацию | [L33](../../../../src/luma_skin_vision/evaluation/__init__.py#L33) |
| `paired_bootstrap` | FunctionDef | См. реализацию | [L57](../../../../src/luma_skin_vision/evaluation/__init__.py#L57) |
| `bootstrap_selective` | FunctionDef | Paired resample subjects, reselect equal image coverage within each resample. | [L75](../../../../src/luma_skin_vision/evaluation/__init__.py#L75) |
| `summarize` | FunctionDef | См. реализацию | [L132](../../../../src/luma_skin_vision/evaluation/__init__.py#L132) |
| `repeatability` | FunctionDef | См. реализацию | [L163](../../../../src/luma_skin_vision/evaluation/__init__.py#L163) |
| `reliability_bins` | FunctionDef | См. реализацию | [L183](../../../../src/luma_skin_vision/evaluation/__init__.py#L183) |
| `segmentation_metrics` | FunctionDef | См. реализацию | [L207](../../../../src/luma_skin_vision/evaluation/__init__.py#L207) |
