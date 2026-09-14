# `scripts/skin_mskcc_vote.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_vote.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched confidence aggregation versus unrolled Huber patch aggregation.

SHA-256 исходника: `29fbd9757079a89e1d7f1941aa044e6dc447175ab2b3b9571158062a296c93bc`. Строк: **32**.

## Зависимости

```python
import torch
from torch import nn
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `PatchVotes` | nn.Module | [L19](../../../../scripts/skin_mskcc_vote.py#L19) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `aggregate` | FunctionDef | См. реализацию | [L6](../../../../scripts/skin_mskcc_vote.py#L6) |
| `PatchVotes` | ClassDef | См. реализацию | [L19](../../../../scripts/skin_mskcc_vote.py#L19) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L20–25</summary>

```python
def __init__(self, target_std, steps=0):
        super().__init__();self.steps=steps
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(768,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,4))
        self.register_buffer('target_std',torch.as_tensor(target_std,dtype=torch.float32))
```

</details>

<details><summary>forward · L27–32</summary>

```python
def forward(self, x, *, details=False):
        h=self.local(x)
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],dim=1))
        vote=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],dim=-1))
        out,risk=aggregate(vote[...,:3],vote[...,3],self.target_std,self.steps)
        return (out,risk,context) if details else out
```

</details>
