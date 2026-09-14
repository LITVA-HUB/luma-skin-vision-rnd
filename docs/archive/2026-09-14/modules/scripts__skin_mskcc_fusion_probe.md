# `scripts/skin_mskcc_fusion_probe.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_fusion_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exploratory fixed50/50fusion of successful CNN and patch models; no fitting.

SHA-256 исходника: `5df5012165b44e7fc133e04037c640e293c3981a05fff509ee1d32647d95a367`. Строк: **36**.

## Зависимости

```python
import json
import numpy as np
from skin_mskcc_data import ROOT
from skin_mskcc_pixels import load
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L10](../../../../scripts/skin_mskcc_fusion_probe.py#L10) |
