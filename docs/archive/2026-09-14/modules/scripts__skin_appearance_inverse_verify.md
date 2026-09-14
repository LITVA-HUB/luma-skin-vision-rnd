# `scripts/skin_appearance_inverse_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_appearance_inverse_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact refits plus independent normal equations, densities and skin metrics.

SHA-256 исходника: `d2d744fff2db443f779ea1ac33a88409166b0e685657fe614cc05240e72834d4`. Строк: **123**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from skin_appearance_inverse import fit_map,predict_map,fit_forward,image_features,forward_means
from skin_appearance_inverse_train import OUT,RUN,PROTOCOL,configs,outputs
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `check_map` | FunctionDef | См. реализацию | [L16](../../../../scripts/skin_appearance_inverse_verify.py#L16) |
| `main` | FunctionDef | См. реализацию | [L29](../../../../scripts/skin_appearance_inverse_verify.py#L29) |
