# `scripts/prepare_cc_v2_fresh.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/prepare_cc_v2_fresh.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Decode the preselected INTEL-TAU subset only after estimator selection lock.

Publisher TIFFs are linear, black corrected and saturation normalized; .wp RGB.
No chart mask is needed: publisher excludes the GT acquisition images.
No CCM, camera identity, target statistics or GT enters image preprocessing.

SHA-256 исходника: `c777cbfea5ecfe630b0931405fe4876e1e3f3bbc818230a0640aa24686d0c16a`. Строк: **138**.

## Зависимости

```python
import argparse
import hashlib
import json
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from download_cc_v2_fresh import FROZEN
from luma_skin_vision.cc.core import experts, unit
from luma_skin_vision.cc.data import sample
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `prepare` | FunctionDef | См. реализацию | [L26](../../../../scripts/prepare_cc_v2_fresh.py#L26) |
