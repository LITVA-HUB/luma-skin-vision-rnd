# `scripts/skin_patch_likelihood_verify.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_patch_likelihood_verify.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Exact conditional-mixture refits, independent densities and native skin error.

SHA-256 исходника: `5f9f1dab293cb0904e5ea69436313da92b9e0b7ef5f27a596b67483b27da47e8`. Строк: **104**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import skin_patch_likelihood as core
from skin_patch_likelihood_train import OUT,RUN,PROTOCOL,STRENGTHS,configs
from skin_appearance_inverse import design
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `main` | FunctionDef | См. реализацию | [L17](../../../../scripts/skin_patch_likelihood_verify.py#L17) |
