# `scripts/skin_expert_anchor.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_expert_anchor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Remove mixture decomposition or explicitly anchor real native-Lab experts.

SHA-256 исходника: `129784a5fecad75d5fbc31f5b371fc1e01fd0f80f59e40e32f78114027d1765d`. Строк: **45**.

## Зависимости

```python
import torch
from torch import nn
from skin_capture_model import CaptureColor
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../scripts/skin_expert_anchor.py#L6)

```python
MECHANISMS=('baseline','plain','uniform_anchor','conditional')
```

[Строка 7](../../../../scripts/skin_expert_anchor.py#L7)

```python
AUGMENTATIONS=('raw','paired')
```

[Строка 8](../../../../scripts/skin_expert_anchor.py#L8)

```python
ARMS=tuple(m+'_'+a for m in MECHANISMS for a in AUGMENTATIONS)
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `PlainColor` | nn.Module | [L11](../../../../scripts/skin_expert_anchor.py#L11) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `PlainColor` | ClassDef | Actual single color head; initial function equals the uniform old model. | [L11](../../../../scripts/skin_expert_anchor.py#L11) |
| `make_model` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_expert_anchor.py#L31) |
| `objective` | FunctionDef | См. реализацию | [L37](../../../../scripts/skin_expert_anchor.py#L37) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L13–21</summary>

```python
def __init__(self,base):
        super().__init__()
        self.local=base.local;self.context=base.context
        last=nn.Linear(256,4)
        with torch.no_grad():
            last.weight[:3].copy_(base.votes[-1].weight[:12].reshape(4,3,256).mean(0))
            last.bias[:3].copy_(base.votes[-1].bias[:12].reshape(4,3).mean(0))
            last.weight[3].copy_(base.votes[-1].weight[12]);last.bias[3].copy_(base.votes[-1].bias[12])
        self.votes=nn.Sequential(base.votes[0],base.votes[1],last)
```

</details>

<details><summary>forward · L23–28</summary>

```python
def forward(self,x):
        h=self.local(x)
        c=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,c[:,None].expand(-1,h.shape[1],-1)],-1))
        p=(v[...,:3]*v[...,3].softmax(1)[...,None]).sum(1)
        return p,p.new_zeros((len(x),4)),p[:,None]
```

</details>
