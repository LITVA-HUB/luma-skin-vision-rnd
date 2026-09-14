# `scripts/skin_image_similarity.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_image_similarity.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Local cross-source image-content screen; TRAIN/VALIDATION only, no biometrics.

SHA-256 исходника: `14a44592819331fa49985b6677670c143b9fb8c598629e37e42c922f6590ca86`. Строк: **223**.

## Зависимости

```python
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import scipy
from PIL import Image
from PIL import __version__ as pillow_version
from scipy.fft import dctn
from threadpoolctl import threadpool_limits
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/skin_image_similarity.py#L19)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 20](../../../../scripts/skin_image_similarity.py#L20)

```python
DATA = Path("D:/Luma-RnD/data_growth_2026_09_14")
```

[Строка 21](../../../../scripts/skin_image_similarity.py#L21)

```python
OUT = DATA / "cross_source_similarity_v1"
```

[Строка 22](../../../../scripts/skin_image_similarity.py#L22)

```python
SEG2 = Path("D:/Luma-RnD/skin_face_transfer_v1/registration.json")
```

[Строка 23](../../../../scripts/skin_image_similarity.py#L23)

```python
SEG2_SHA = "c92d12e53e42325cec76fd6ad61dc94a8c685fcba9a48b97ccc0a791d108da88"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `digest` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_image_similarity.py#L26) |
| `read` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_image_similarity.py#L31) |
| `write_once` | FunctionDef | См. реализацию | [L35](../../../../scripts/skin_image_similarity.py#L35) |
| `describe` | FunctionDef | См. реализацию | [L47](../../../../scripts/skin_image_similarity.py#L47) |
| `candidates` | FunctionDef | См. реализацию | [L61](../../../../scripts/skin_image_similarity.py#L61) |
| `similarity` | FunctionDef | См. реализацию | [L74](../../../../scripts/skin_image_similarity.py#L74) |
| `sources` | FunctionDef | См. реализацию | [L85](../../../../scripts/skin_image_similarity.py#L85) |
| `search` | FunctionDef | См. реализацию | [L120](../../../../scripts/skin_image_similarity.py#L120) |
| `main` | FunctionDef | См. реализацию | [L142](../../../../scripts/skin_image_similarity.py#L142) |
