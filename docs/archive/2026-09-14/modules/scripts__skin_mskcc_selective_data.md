# `scripts/skin_mskcc_selective_data.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_selective_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Decode reserved roles only after an exact verified pre-calibration/final lock.

SHA-256 исходника: `d7429171ee96f52abbc863d9c4cea10358f19205335f28bfc1842184f5c1fc9d`. Строк: **60**.

## Зависимости

```python
import concurrent.futures
import json
import numpy as np
from skin_mskcc_data import ROOT,RAW,read,manifest,sha
from skin_mskcc_pixels import decode
from skin_mskcc_selective_core import OUT,verify_lock
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `instrument_reference` | FunctionDef | Mean of available COMPLETE instrument readings, checked against author mean.  Missing repetitions stay NaN for provenance; no reference is imputed. See the pre-test reference-completeness amendment. | [L10](../../../../scripts/skin_mskcc_selective_data.py#L10) |
| `sealed` | FunctionDef | См. реализацию | [L32](../../../../scripts/skin_mskcc_selective_data.py#L32) |
