# `scripts/skin_spatial_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_spatial_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched conventional spatial processing and anchored latent graph diffusion.

SHA-256 исходника: `42f9881376f2f1a6ff3948aad09e6353112fe37cec9b19ed460a0010558d6d86`. Строк: **72**.

## Зависимости

```python
import torch
from torch import nn
from torch.nn import functional as F
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../scripts/skin_spatial_model.py#L6)

```python
ARMS=['plain','conv1','conv3','graph1','graph3','graph3_scrambled']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `SpatialColor` | nn.Module | [L25](../../../../scripts/skin_spatial_model.py#L25) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `grid_adjacency` | FunctionDef | См. реализацию | [L9](../../../../scripts/skin_spatial_model.py#L9) |
| `screened_diffusion` | FunctionDef | Jacobi iteration for (diag(anchor)+L_weights) z=anchor*h.  Positive anchors and nonnegative weights give a convex-combination update. Finite steps approximate this quadratic latent problem, not skin physics. | [L14](../../../../scripts/skin_spatial_model.py#L14) |
| `SpatialColor` | ClassDef | См. реализацию | [L25](../../../../scripts/skin_spatial_model.py#L25) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L26–43</summary>

```python
def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.steps=0 if arm=='plain' else (1 if arm.endswith('1') else 3)
        # Core initialized first and identically to the historical plain PatchVotes.
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(768,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,4))
        # All arms store the same modules, with active counts reported separately.
        self.relation=nn.Linear(256,256,bias=False)
        self.anchor=nn.Linear(256,1)
        self.temperature=nn.Parameter(torch.tensor(0.))
        self.coupling=nn.Parameter(torch.tensor(0.))
        self.spatial=nn.Conv2d(256,256,3,padding=1,groups=256)
        self.register_buffer('adjacency',grid_adjacency())
        permutation=torch.randperm(64,generator=torch.Generator().manual_seed(20260911))
        self.register_buffer('permutation',permutation)
        self.register_buffer('inverse_permutation',torch.argsort(permutation))
```

</details>

<details><summary>forward · L53–72</summary>

```python
def forward(self,x):
        h=self.local(x)
        if self.steps:
            if self.arm.startswith('graph'):
                base=h[:,self.permutation] if self.arm.endswith('scrambled') else h
                square=base.square().mean(-1)
                distances=(square[:,:,None]+square[:,None,:]-2*(base@base.transpose(1,2))/256).clamp_min(0)
                weights=self.adjacency*torch.exp(-distances/(F.softplus(self.temperature)+.001))*F.softplus(self.coupling)
                anchor=F.softplus(self.anchor(base))+.05
                residual=screened_diffusion(base,anchor,weights,self.steps)-base
                if self.arm.endswith('scrambled'):residual=residual[:,self.inverse_permutation]
            else:
                z=h.transpose(1,2).reshape(-1,256,8,8)
                for _ in range(self.steps):z=z+.25*F.silu(self.spatial(z))
                residual=z.flatten(2).transpose(1,2)-h
            h=h+.5*torch.tanh(self.relation(residual))
        context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,64,-1)],-1))
        weight=v[...,3].softmax(1);color=(v[...,:3]*weight[...,None]).sum(1)
        return color,v[...,:3],weight
```

</details>
