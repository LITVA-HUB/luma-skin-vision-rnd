# `scripts/skin_appearance_inverse_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_appearance_inverse_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen closed-form forward/inverse skin color experiment, source roles only.

SHA-256 исходника: `2244bff6f220885be598d69d122dbd78197a68b75bac830c6180a0f062c9b3b7`. Строк: **96**.

## Зависимости

```python
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_appearance_inverse import image_features,fit_map,predict_map,fit_forward,infer_forward
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_pair_train import subset,write,rows
from skin_distribution_train import score
from skin_mskcc_summary_pilot import summarize
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/skin_appearance_inverse_train.py#L14)

```python
OUT=ROOT/'docs/benchmarks/skin_appearance_inverse_v1'
```

[Строка 15](../../../../scripts/skin_appearance_inverse_train.py#L15)

```python
RUN=ROOT/'experiments/runs/skin_appearance_inverse_v1'
```

[Строка 16](../../../../scripts/skin_appearance_inverse_train.py#L16)

```python
PROTOCOL=ROOT/'docs/research/skin_appearance_inverse_protocol_v1.md'
```

[Строка 17](../../../../scripts/skin_appearance_inverse_train.py#L17)

```python
ALPHAS=(.01,1.,100.)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `configs` | FunctionDef | См. реализацию | [L20](../../../../scripts/skin_appearance_inverse_train.py#L20) |
| `novelty` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_appearance_inverse_train.py#L27) |
| `outputs` | FunctionDef | См. реализацию | [L32](../../../../scripts/skin_appearance_inverse_train.py#L32) |
| `run_fit` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_appearance_inverse_train.py#L38) |
| `main` | FunctionDef | См. реализацию | [L77](../../../../scripts/skin_appearance_inverse_train.py#L77) |
