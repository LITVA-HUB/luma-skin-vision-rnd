# `scripts/skin_risk_cross.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_risk_cross.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed two-model crossing of real skin color and Gaussian risk.

SHA-256 исходника: `61c4969df0d6bc82c93f7432d69fcc5c4f33d22a8d8fa9ff6a2be3abdc6fd747`. Строк: **94**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from luma_skin_vision.color import delta_e00
from skin_distribution_train import OUT as SOURCE,RUN as SOURCE_RUN,score
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/skin_risk_cross.py#L14)

```python
OUT=ROOT/'docs/benchmarks/skin_risk_cross_v1'
```

[Строка 15](../../../../scripts/skin_risk_cross.py#L15)

```python
RUN=ROOT/'experiments/runs/skin_risk_cross_v1'
```

[Строка 16](../../../../scripts/skin_risk_cross.py#L16)

```python
PROTOCOL=ROOT/'docs/research/skin_risk_cross_protocol_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `expected_error` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_risk_cross.py#L19) |
| `main` | FunctionDef | См. реализацию | [L30](../../../../scripts/skin_risk_cross.py#L30) |
