# `scripts/document_cc_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/document_cc_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Build human-readable CC v2 tables from frozen per-image evaluation records.

SHA-256 исходника: `a5e37d99f226df9f2ee1911977b74c21fcd59615cfdfc3e835ce11666edc2d1d`. Строк: **285**.

## Зависимости

```python
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/document_cc_v2.py#L12)

```python
ROOT = Path("docs/benchmarks/cc_v2")
```

[Строка 13](../../../../scripts/document_cc_v2.py#L13)

```python
PRIMARY = "ccv2_sog_large_g0::combined"
```

[Строка 14](../../../../scripts/document_cc_v2.py#L14)

```python
MATCHED = "ccv2_direct_large_g0::combined"
```

[Строка 15](../../../../scripts/document_cc_v2.py#L15)

```python
METHODS = {
    PRIMARY: "Proposed: SoG residual, combined risk",
    MATCHED: "Strong matched C+: direct, combined risk",
    "ccv2_direct_large_g0::context": "Standard C+: direct, context risk",
    "ccv2_legacy_proposed::upgraded_combined": "V1 mixture, upgraded risk",
    "ccv2_legacy_baseline::upgraded_combined": "V1 direct, upgraded risk",
    "direct_hgb7::combined": "Statistics + HGB, combined risk (1 fit)",
    "gw_ridge1::combined": "GW + ridge, combined risk (1 fit)",
    "gw_ridge1::cheap": "GW + ridge, cheap risk (ablation, 1 fit)",
    "ccv2_legacy_baseline::v1_gray_world": "Gray World + frozen V1 selector",
    "ccv2_legacy_baseline::v1_max_rgb": "Max RGB + frozen V1 selector",
    "ccv2_legacy_baseline::v1_shades_gray": "Shades of Gray + frozen V1 selector",
    "ccv2_legacy_baseline::v1_gray_edge": "Gray Edge + frozen V1 selector",
    "ccv2_gw_large_g0::combined": "GW residual, combined (exploratory, 1 seed)",
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `table` | FunctionDef | См. реализацию | [L32](../../../../scripts/document_cc_v2.py#L32) |
| `main` | FunctionDef | См. реализацию | [L39](../../../../scripts/document_cc_v2.py#L39) |
