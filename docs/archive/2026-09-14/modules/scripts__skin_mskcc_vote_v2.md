# `scripts/skin_mskcc_vote_v2.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_vote_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source ablations: prevent global-vote shortcut and activate robust weighting.

SHA-256 исходника: `92422d94006c5ab07d11974e896524bf986afe1cf2f90f2c9b785055b7bef9b6`. Строк: **38**.

## Зависимости

```python
import torch
from torch import nn
from skin_mskcc_vote import PatchVotes
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `GlobalColorMLP` | nn.Module | [L7](../../../../scripts/skin_mskcc_vote_v2.py#L7) |
| `VoteAblation` | PatchVotes | [L19](../../../../scripts/skin_mskcc_vote_v2.py#L19) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `GlobalColorMLP` | ClassDef | См. реализацию | [L7](../../../../scripts/skin_mskcc_vote_v2.py#L7) |
| `VoteAblation` | ClassDef | См. реализацию | [L19](../../../../scripts/skin_mskcc_vote_v2.py#L19) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L8–12</summary>

```python
def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(36,512),nn.SiLU(),nn.Linear(512,768),nn.SiLU(),
                               nn.Linear(768,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU())
        self.head=nn.Linear(256,3)
```

</details>

<details><summary>forward · L14–16</summary>

```python
def forward(self,x,*,details=False):
        h=self.body(x);out=self.head(h)
        return (out,torch.zeros(len(x),device=x.device),h) if details else out
```

</details>

<details><summary>__init__ · L20–21</summary>

```python
def __init__(self,target_std,*,local_only,steps):
        super().__init__(target_std,steps);self.local_only=local_only
```

</details>

<details><summary>forward · L23–38</summary>

```python
def forward(self,x,*,details=False):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        broadcast=context[:,None].expand(-1,h.shape[1],-1)
        prediction=self.votes(torch.cat([h,broadcast],-1))
        local=(self.votes(torch.cat([h,torch.zeros_like(broadcast)],-1))[...,:3]
               if self.local_only else prediction[...,:3])
        base=prediction[...,3].softmax(1);weight=base
        out=(local*weight[...,None]).sum(1)
        for _ in range(self.steps):
            residual=torch.linalg.vector_norm((local-out[:,None])*self.target_std,dim=-1)
            weight=base*(.5/residual.clamp_min(1e-6)).clamp(max=1.)
            weight=weight/weight.sum(1,keepdim=True).clamp_min(1e-12)
            out=(local*weight[...,None]).sum(1)
        risk=(((local-out[:,None])*self.target_std).square().sum(-1)*weight).sum(1).clamp_min(0).sqrt()
        return (out,risk,context) if details else out
```

</details>
