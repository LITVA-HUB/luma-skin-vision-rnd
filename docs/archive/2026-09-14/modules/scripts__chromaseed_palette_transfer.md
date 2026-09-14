# `scripts/chromaseed_palette_transfer.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

P3 initialization: change the local encoder, preserve the native HR model.

SHA-256 исходника: `96ec09487640a96cd2b1ee81654998fdb2cbf29f6ae6bb909e07c239b281d30d`. Строк: **77**.

## Зависимости

```python
from __future__ import annotations
import hashlib
import numpy as np
import torch
from chromaseed_architecture_scale import SEEDS, specs
from chromaseed_head_range import Bank as NativeBank
from chromaseed_palette_encoder import ENCODER_PARAMETERS, transplant
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 13](../../../../scripts/chromaseed_palette_transfer.py#L13)

```python
VARIANTS = ("patch5m", "soft5m", "dynamic5m")
```

[Строка 14](../../../../scripts/chromaseed_palette_transfer.py#L14)

```python
ARMS = ("original", "aligned", "shuffled")
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Bank` | NativeBank | [L53](../../../../scripts/chromaseed_palette_transfer.py#L53) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `encoder_digest` | FunctionDef | См. реализацию | [L17](../../../../scripts/chromaseed_palette_transfer.py#L17) |
| `validate_encoders` | FunctionDef | См. реализацию | [L28](../../../../scripts/chromaseed_palette_transfer.py#L28) |
| `Bank` | ClassDef | См. реализацию | [L53](../../../../scripts/chromaseed_palette_transfer.py#L53) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L54–70</summary>

```python
def __init__(self, variant, mode, arm, encoders=None, mean=None, std=None):
        if variant not in VARIANTS:
            raise ValueError("P3 requires an architecture with the registered 18→384→256 encoder")
        validate_encoders(encoders, arm)
        assert specs(variant)[:2] == [("token1", 18, 384), ("token2", 384, 256)]
        super().__init__(variant, mode)
        self.palette_arm = arm
        self.encoder_digests = []
        if arm == "original":
            return
        translated = [transplant(model, mean, std) for model in encoders]
        self.encoder_digests = [encoder_digest(m) for m in encoders]
        with torch.no_grad():
            for slot in range(6):
                self.theta[slot, :ENCODER_PARAMETERS].copy_(
                    torch.from_numpy(translated[slot // 2]["theta"])
                )
```

</details>

<details><summary>export · L72–77</summary>

```python
def export(self, slot, warm, tokens):
        model = super().export(slot, warm, tokens)
        if self.palette_arm != "original":
            model["palette_arm"] = np.asarray(self.palette_arm)
            model["palette_encoder_digest"] = np.asarray(self.encoder_digests[slot // 2])
        return model
```

</details>
