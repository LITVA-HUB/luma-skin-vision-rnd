# `tests/test_skin_capture_support.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_capture_support.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `67ac1dcc4c836edda14eb2ade34bf4c95f24a0b522845b5fd9ff2cbbedc7e14d`. Строк: **47**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
from skin_capture_support import make_plan,apply_plan
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_bag_elements_are_exact_observations_and_modes_count_origins` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_capture_support.py#L9) |
| `test_no_augmentation_and_self_bootstrap_keep_original_mode` | FunctionDef | См. реализацию | [L23](../../../../tests/test_skin_capture_support.py#L23) |
| `test_seeded_plans_repeat_exactly` | FunctionDef | См. реализацию | [L35](../../../../tests/test_skin_capture_support.py#L35) |
| `test_soft_mode_control_matches_union_labels_without_changing_patches` | FunctionDef | См. реализацию | [L40](../../../../tests/test_skin_capture_support.py#L40) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_bag_elements_are_exact_observations_and_modes_count_origins` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bag_elements_are_exact_observations_and_modes_count_origins():
    x=torch.arange(4*64*18,dtype=torch.float32).reshape(4,64,18)
    modes=torch.eye(4);partner=torch.tensor([2,3,0,1])
    plan=make_plan(4,64,np.random.default_rng(17));plan['augment'][:]=True
    for arm in ('paired_union','paired_stratified'):
        out,target,origin,index=apply_plan(x,modes,partner,plan,arm)
        for i in range(4):
            for j in range(64):
                source=int(partner[i]) if origin[i,j] else i
                torch.testing.assert_close(out[i,j],x[source,index[i,j]],rtol=0,atol=0)
            fraction=origin[i].float().mean()
            torch.testing.assert_close(target[i],modes[i]*(1-fraction)+modes[partner[i]]*fraction,rtol=0,atol=0)
```

</details>

### `test_no_augmentation_and_self_bootstrap_keep_original_mode` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_no_augmentation_and_self_bootstrap_keep_original_mode():
    x=torch.rand(4,64,18);modes=torch.eye(4);partner=torch.tensor([2,3,0,1])
    plan=make_plan(4,64,np.random.default_rng(31));plan['augment'][:]=False
    for arm in ('baseline','self_bootstrap','paired_union','paired_stratified'):
        out,target,_,_=apply_plan(x,modes,partner,plan,arm)
        torch.testing.assert_close(out,x,rtol=0,atol=0);torch.testing.assert_close(target,modes,rtol=0,atol=0)
    plan['augment'][:]=True
    out,target,origin,index=apply_plan(x,modes,partner,plan,'self_bootstrap')
    assert not origin.any();torch.testing.assert_close(target,modes,rtol=0,atol=0)
    torch.testing.assert_close(out,x.gather(1,index[...,None].expand(-1,-1,18)),rtol=0,atol=0)
```

</details>

### `test_seeded_plans_repeat_exactly` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_seeded_plans_repeat_exactly():
    a=make_plan(32,64,np.random.default_rng(314));b=make_plan(32,64,np.random.default_rng(314))
    for k in a:np.testing.assert_array_equal(a[k],b[k])
```

</details>

### `test_soft_mode_control_matches_union_labels_without_changing_patches` · L40

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_soft_mode_control_matches_union_labels_without_changing_patches():
    x=torch.rand(4,64,18);modes=torch.eye(4);partner=torch.tensor([2,3,0,1])
    plan=make_plan(4,64,np.random.default_rng(17));plan['augment'][:]=True
    original,target,origin,index=apply_plan(x,modes,partner,plan,'soft_mode_control')
    _,mixed_target,_,_=apply_plan(x,modes,partner,plan,'paired_union')
    torch.testing.assert_close(original,x,rtol=0,atol=0)
    torch.testing.assert_close(target,mixed_target,rtol=0,atol=0)
    assert not origin.any()
```

</details>
