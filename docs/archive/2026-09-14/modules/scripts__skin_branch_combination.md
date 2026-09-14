# `scripts/skin_branch_combination.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_branch_combination.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed equal-weight combinations; no blend fitting, no new endpoint access.

SHA-256 исходника: `41808e0dab104cb49cbd97c0fa791ec1a1fb7a3e1fe020ac03cb0be3f1f5c116`. Строк: **95**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_branch_combination.py#L9)

```python
OUT=ROOT/'docs/benchmarks/skin_branch_combination_v1'
```

[Строка 10](../../../../scripts/skin_branch_combination.py#L10)

```python
PROTOCOL=ROOT/'docs/research/skin_branch_combination_protocol_v1.md'
```

[Строка 11](../../../../scripts/skin_branch_combination.py#L11)

```python
PAIRS={
 'mixture_graph_always':(('skin_capture_v1','mixture_mse'),('skin_train_branch_v1','graph_always')),
 'mixture_graph_drop':(('skin_capture_v1','mixture_mse'),('skin_train_branch_v1','graph_drop')),
 'mixture_conv_drop':(('skin_capture_v1','mixture_mse'),('skin_train_branch_v1','conv_drop')),
 'mixture_plain':(('skin_capture_v1','mixture_mse'),('skin_capture_v1','plain_mse')),
 'plain_graph_always':(('skin_capture_v1','plain_mse'),('skin_train_branch_v1','graph_always'))}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `metrics` | FunctionDef | См. реализацию | [L19](../../../../scripts/skin_branch_combination.py#L19) |
| `main` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_branch_combination.py#L24) |
