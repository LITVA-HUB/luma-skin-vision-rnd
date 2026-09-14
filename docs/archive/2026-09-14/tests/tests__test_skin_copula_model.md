# `tests/test_skin_copula_model.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_copula_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `319703fd4c1333303f1d350327a769c13302aeb953dfd05d8daef846683119b2`. Строк: **29**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
from skin_copula_model import DistributionColor,ARMS
from skin_copula_data import pack
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_pack_keeps_absolute_tokens_except_declared_ablation` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_copula_model.py#L10) |
| `test_exact_matching_capacity_and_feature_context_effect` | FunctionDef | См. реализацию | [L19](../../../../tests/test_skin_copula_model.py#L19) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_pack_keeps_absolute_tokens_except_declared_ablation` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pack_keeps_absolute_tokens_except_declared_ablation():
    d={'tokens':np.ones((2,64,18)),'rank_tokens':np.zeros((2,64,18)),
       'hist':np.ones((2,512))/512,'rank_hist':np.eye(512)[:2]}
    for arm in ['none','rgb_hist','copula']:
        np.testing.assert_array_equal(pack(d,arm)[:,:1152],d['tokens'].reshape(2,-1))
    assert not pack(d,'rank_only')[:,:1152].any()
    assert not pack(d,'none')[:,1152:].any()
```

</details>

### `test_exact_matching_capacity_and_feature_context_effect` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_exact_matching_capacity_and_feature_context_effect():
    states=[];counts=[]
    for arm in ARMS:
        torch.manual_seed(83);m=DistributionColor(arm);states.append(m.state_dict());counts.append(sum(p.numel() for p in m.parameters()))
    assert len(set(counts))==1
    for state in states[1:]:
        for k,v in state.items():assert torch.equal(v,states[0][k])
    x=torch.rand(2,1664);x[:,1152:]=0;x[0,1152]=1;x[1,-1]=1;x[1,:1152]=x[0,:1152]
    p,_,_=m(x);assert not torch.equal(p[0],p[1])
    p.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
```

</details>
