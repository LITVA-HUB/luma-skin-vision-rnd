# `scripts/skin_correspondence.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_correspondence.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-only measurement/capture diagnostics; no fitting or new test access.

SHA-256 исходника: `2333baec935615304bc6c6e28216ec04d871ba0e8ffd04ca27e1696cd77eddd2`. Строк: **169**.

## Зависимости

```python
import itertools
import json
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../scripts/skin_correspondence.py#L11)

```python
OUT = ROOT / 'docs/benchmarks/skin_correspondence_v1'
```

[Строка 12](../../../../scripts/skin_correspondence.py#L12)

```python
PROTOCOL = ROOT / 'docs/research/skin_correspondence_protocol_v1.md'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `unique_references` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_correspondence.py#L15) |
| `site_decomposition` | FunctionDef | См. реализацию | [L30](../../../../scripts/skin_correspondence.py#L30) |
| `stats` | FunctionDef | См. реализацию | [L55](../../../../scripts/skin_correspondence.py#L55) |
| `balanced` | FunctionDef | См. реализацию | [L61](../../../../scripts/skin_correspondence.py#L61) |
| `main` | FunctionDef | См. реализацию | [L68](../../../../scripts/skin_correspondence.py#L68) |
