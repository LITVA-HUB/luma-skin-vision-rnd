# `scripts/skin_capture_model.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_capture_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched latent-capture color hypotheses and instrument perceptual loss.

SHA-256 исходника: `23c83500f35cc44b379597451f117ae37ad85fe30ea872a480aa159ea8d015cf`. Строк: **62**.

## Зависимости

```python
import torch
from torch import nn
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../scripts/skin_capture_model.py#L5)

```python
MODES=['NP-C','NP-NC','P-C','P-NC']
```

[Строка 6](../../../../scripts/skin_capture_model.py#L6)

```python
ARCHES=['plain','uniform','mixture']
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `CaptureColor` | nn.Module | [L43](../../../../scripts/skin_capture_model.py#L43) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `_root` | FunctionDef | См. реализацию | [L9](../../../../scripts/skin_capture_model.py#L9) |
| `_hue` | FunctionDef | См. реализацию | [L14](../../../../scripts/skin_capture_model.py#L14) |
| `delta_e00_squared` | FunctionDef | CIEDE2000 squared, kL=kC=kH=1, computed in double for hue boundaries.  Autograd is piecewise valid; this does not make the original formula globally smooth. At exact zero chroma/root arguments use a finite zero convention. | [L20](../../../../scripts/skin_capture_model.py#L20) |
| `CaptureColor` | ClassDef | All arms share exact parameter shapes; camera/mode is never an input. | [L43](../../../../scripts/skin_capture_model.py#L43) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>__init__ · L45–52</summary>

```python
def __init__(self,arch):
        super().__init__()
        if arch not in ARCHES:raise ValueError('Unknown architecture')
        self.arch=arch
        self.local=nn.Sequential(nn.Linear(18,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.context=nn.Sequential(nn.Linear(768,512),nn.SiLU(),nn.Linear(512,512),nn.SiLU())
        self.votes=nn.Sequential(nn.Linear(768,256),nn.SiLU(),nn.Linear(256,13))
        self.gate=nn.Linear(512,4)
```

</details>

<details><summary>forward · L54–62</summary>

```python
def forward(self,x):
        h=self.local(x);context=self.context(torch.cat([h.mean(1),h.amax(1),h.std(1,correction=0)],1))
        v=self.votes(torch.cat([h,context[:,None].expand(-1,h.shape[1],-1)],-1))
        weights=v[...,12].softmax(1)
        hypotheses=(v[...,:12].reshape(len(x),h.shape[1],4,3)*weights[:,:,None,None]).sum(1)
        logits=self.gate(context);gate=logits.softmax(1)
        coefficient=gate if self.arch=='mixture' else torch.full_like(gate,.25)
        prediction=(hypotheses*coefficient[...,None]).sum(1)
        return prediction,logits,hypotheses
```

</details>
