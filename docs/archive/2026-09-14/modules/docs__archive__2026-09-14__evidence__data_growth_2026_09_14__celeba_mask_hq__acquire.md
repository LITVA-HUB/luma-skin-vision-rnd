# `docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/acquire.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/acquire.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Download the author-linked public archive; preserve original bytes and receipt.

SHA-256 исходника: `de2f6e82f657bf7a6acaddc2ef108fef2ae0b25b8850166d9c878ff556f53cc2`. Строк: **65**.

## Зависимости

```python
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from skin_lapa_acquire import DownloadForm
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/acquire.py#L11)

```python
ROOT = Path(__file__).resolve().parent
```

[Строка 12](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/acquire.py#L12)

```python
SCRIPTS = Path('C:/Users/dimal/Documents/просто/.worktrees/luma-local-search/scripts')
```

[Строка 16](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/acquire.py#L16)

```python
ARCHIVE = ROOT / 'CelebAMask-HQ.zip'
```

[Строка 17](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/acquire.py#L17)

```python
PARTIAL = ROOT / 'CelebAMask-HQ.zip.partial'
```

[Строка 18](../../../../docs/archive/2026-09-14/evidence/data_growth_2026_09_14/celeba_mask_hq/acquire.py#L18)

```python
RECEIPT = ROOT / 'archive_source.json'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
