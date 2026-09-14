# `scripts/skin_mskcc_pixels.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_pixels.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-only verified pixel cache and deterministic color/patch features.

SHA-256 исходника: `c5b671ac38da0bec62c18c3872fbac52f2f8a239f0e155a74c44f8ec1b91ad0a`. Строк: **80**.

## Зависимости

```python
import concurrent.futures
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from skin_mskcc_data import ROOT, RAW, MANIFEST, sha, load_source
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 9](../../../../scripts/skin_mskcc_pixels.py#L9)

```python
PROTOCOL = ROOT/'docs/research/skin_mskcc_pixel_protocol_v1.md'
```

[Строка 10](../../../../scripts/skin_mskcc_pixels.py#L10)

```python
CACHE = ROOT/'data/processed/skin_mskcc_pixels_v1'
```

[Строка 11](../../../../scripts/skin_mskcc_pixels.py#L11)

```python
BENCH = ROOT/'docs/benchmarks/skin_mskcc_pixels_v1'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `features` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_mskcc_pixels.py#L14) |
| `decode` | FunctionDef | См. реализацию | [L40](../../../../scripts/skin_mskcc_pixels.py#L40) |
| `main` | FunctionDef | См. реализацию | [L51](../../../../scripts/skin_mskcc_pixels.py#L51) |
| `load` | FunctionDef | См. реализацию | [L72](../../../../scripts/skin_mskcc_pixels.py#L72) |
