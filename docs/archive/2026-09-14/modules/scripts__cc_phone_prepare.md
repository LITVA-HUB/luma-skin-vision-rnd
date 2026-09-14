# `scripts/cc_phone_prepare.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_prepare.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Prepare only frozen phone TRAIN loader scenes under reference protocol v1.

SHA-256 исходника: `bfd4e364ba0797dfbdff3db5bc94d68c10df467211a7db79320fd5a1e167820b`. Строк: **109**.

## Зависимости

```python
import argparse
import json
from pathlib import Path
import numpy as np
from cc_phone_loader_audit import LOCK, ROOT, loader_scenes, open_camera_rgb, patch_stats
from luma_skin_vision.cc.core import EXPERT_NAMES, angular, experts, reproduction, summarize
from luma_skin_vision.cc.data import sample
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 14](../../../../scripts/cc_phone_prepare.py#L14)

```python
PROTOCOL = ROOT/'docs/research/phone_reference_protocol_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `reference_from_patches` | FunctionDef | См. реализацию | [L17](../../../../scripts/cc_phone_prepare.py#L17) |
| `prepare` | FunctionDef | См. реализацию | [L43](../../../../scripts/cc_phone_prepare.py#L43) |
