# `scripts/skin_face_segment.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_face_segment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

ChromaSeed-Seg1: spatial facial-skin masks, separate from instrument color regression.

SHA-256 исходника: `e66589cdf8ab369e6754da9292df36bdfd6413754a2dd2d65c50f3c504f4cff9`. Строк: **77**.

## Зависимости

```python
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `ConvBlock` | nn.Sequential | [L12](../../../../scripts/skin_face_segment.py#L12) |
| `SkinUNet` | nn.Module | [L20](../../../../scripts/skin_face_segment.py#L20) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `ConvBlock` | ClassDef | См. реализацию | [L12](../../../../scripts/skin_face_segment.py#L12) |
| `SkinUNet` | ClassDef | См. реализацию | [L20](../../../../scripts/skin_face_segment.py#L20) |
| `binary_mask` | FunctionDef | См. реализацию | [L45](../../../../scripts/skin_face_segment.py#L45) |
| `skin_loss` | FunctionDef | См. реализацию | [L52](../../../../scripts/skin_face_segment.py#L52) |
| `scores` | FunctionDef | См. реализацию | [L60](../../../../scripts/skin_face_segment.py#L60) |
| `augment` | FunctionDef | См. реализацию | [L71](../../../../scripts/skin_face_segment.py#L71) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L13–17</summary>

```python
def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),
                         nn.GroupNorm(math.gcd(8,cout),cout),nn.SiLU(),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),
                         nn.GroupNorm(math.gcd(8,cout),cout),nn.SiLU())
```

</details>

<details><summary>__init__ · L21–28</summary>

```python
def __init__(self,width=24):
        super().__init__()
        self.width = width
        widths = [width*2**i for i in range(5)]
        self.encoder = nn.ModuleList([ConvBlock(3,widths[0])] +
                                     [ConvBlock(a,b) for a,b in zip(widths[:-1],widths[1:],strict=True)])
        self.decoder = nn.ModuleList([ConvBlock(widths[i]+widths[i-1],widths[i-1]) for i in range(4,0,-1)])
        self.output = nn.Conv2d(width,1,1)
```

</details>

<details><summary>forward · L30–42</summary>

```python
def forward(self,x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] % 16 or x.shape[3] % 16:
            raise ValueError('RGB spatial dimensions must be a multiple of 16')
        skips = []
        for i,block in enumerate(self.encoder):
            if i:
                x = F.max_pool2d(x,2)
            x = block(x)
            skips.append(x)
        for block,skip in zip(self.decoder,reversed(skips[:-1]),strict=True):
            x = F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x = block(torch.cat((x,skip),dim=1))
        return self.output(x)
```

</details>

<details><summary>skin_loss · L52–57</summary>

```python
def skin_loss(logits,target):
    logits = logits.float()
    probability = torch.sigmoid(logits)
    axes = (1,2,3)
    dice = (2*(probability*target).sum(axes)+1)/(probability.sum(axes)+target.sum(axes)+1)
    return F.binary_cross_entropy_with_logits(logits,target)+(1-dice).mean()
```

</details>
