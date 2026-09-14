# `scripts/skin_mskcc_selective_metrics.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_selective_metrics.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Explicit empirical color risk and calibration-threshold evaluation.

SHA-256 исходника: `211eedcb3ad05980637523e7fd4221e808eda96e31452dbe3fd8f2a989b75ae7`. Строк: **38**.

## Зависимости

```python
import numpy as np
from skin_mskcc_summary_pilot import summarize
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_mskcc_selective_metrics.py#L5)

```python
COVERAGES=[1,.95,.9,.8,.7,.6]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `rows` | FunctionDef | См. реализацию | [L8](../../../../scripts/skin_mskcc_selective_metrics.py#L8) |
| `curve` | FunctionDef | См. реализацию | [L12](../../../../scripts/skin_mskcc_selective_metrics.py#L12) |
| `selection` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_mskcc_selective_metrics.py#L17) |
| `evaluate` | FunctionDef | См. реализацию | [L22](../../../../scripts/skin_mskcc_selective_metrics.py#L22) |
| `accepted` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_mskcc_selective_metrics.py#L34) |
