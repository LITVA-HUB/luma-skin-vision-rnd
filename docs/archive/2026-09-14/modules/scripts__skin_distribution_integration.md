# `scripts/skin_distribution_integration.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_distribution_integration.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Post-hoc fixed antithetic Sobol integration; no fitting or held-out loading.

SHA-256 исходника: `7947737606cf7e79e786c58eae3e84f739c6e8bded34d0563d58881266efb39f`. Строк: **88**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from luma_skin_vision.color import delta_e00
from skin_distribution_train import OUT,RUN,score
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import subset,write
from skin_mskcc_audit import scalar_de
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/skin_distribution_integration.py#L14)

```python
PROTOCOL=ROOT/'docs/research/skin_distribution_integration_diagnostic_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `integrate` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_distribution_integration.py#L17) |
| `main` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_distribution_integration.py#L38) |
