# `scripts/skin_mskcc_color_registry.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_color_registry.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

All frozen local-image color comparators, including negative source arms.

SHA-256 исходника: `35de560d81c34db0389da6da51d63f83f11ec3aa84418478d0a1b1c3b734a711`. Строк: **45**.

## Зависимости

```python
import joblib
import numpy as np
import torch
from torchvision.models import mobilenet_v3_small
from skin_mskcc_data import ROOT
from skin_mskcc_vote import PatchVotes
from skin_mskcc_vote_v2 import GlobalColorMLP,VoteAblation
from skin_mskcc_train_pixels import predict
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/skin_mskcc_color_registry.py#L11)

```python
ARCHES=['cnn','votes_mean','votes_huber3','global_mlp','shared_tight','local_mean','local_tight']
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `checkpoint` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_mskcc_color_registry.py#L14) |
| `paths` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_mskcc_color_registry.py#L19) |
| `all_predictions` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_mskcc_color_registry.py#L24) |
