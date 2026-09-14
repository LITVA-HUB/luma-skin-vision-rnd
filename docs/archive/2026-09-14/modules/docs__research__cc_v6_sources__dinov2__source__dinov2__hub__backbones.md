# `docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `871fca671b12a9ff02e810654baf509e97ccf461bf8196ce5ddeefff2fd87d3e`. Строк: **172**.

## Зависимости

```python
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse
from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name, _safe_load_state_dict_from_url
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Weights` | Enum | [L14](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L14) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Weights` | ClassDef | См. реализацию | [L14](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L14) |
| `is_url` | FunctionDef | См. реализацию | [L19](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L19) |
| `convert_path_or_url_to_url` | FunctionDef | См. реализацию | [L24](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L24) |
| `_make_dinov2_model` | FunctionDef | См. реализацию | [L30](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L30) |
| `dinov2_vits14` | FunctionDef | DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset. | [L80](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L80) |
| `dinov2_vitb14` | FunctionDef | DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset. | [L87](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L87) |
| `dinov2_vitl14` | FunctionDef | DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset. | [L94](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L94) |
| `dinov2_vitg14` | FunctionDef | DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset. | [L101](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L101) |
| `dinov2_vits14_reg` | FunctionDef | DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset. | [L114](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L114) |
| `dinov2_vitb14_reg` | FunctionDef | DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset. | [L129](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L129) |
| `dinov2_vitl14_reg` | FunctionDef | DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset. | [L144](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L144) |
| `dinov2_vitg14_reg` | FunctionDef | DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset. | [L159](../../../../docs/research/cc_v6_sources/dinov2/source/dinov2/hub/backbones.py#L159) |
