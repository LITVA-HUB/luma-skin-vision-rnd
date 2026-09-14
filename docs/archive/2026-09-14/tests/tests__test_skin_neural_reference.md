# `tests/test_skin_neural_reference.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_neural_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent solver, exclusion, gradient and capacity checks for color adapters.

SHA-256 исходника: `82e2e9eb053627792d2872240517156576c31d0a4b0b0169d54b34ba63235742`. Строк: **63**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_weighted_solver_matches_augmented_lstsq_and_excludes_people` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_neural_reference.py#L10) |
| `test_solver_gradients_and_reference_mean_constant_limit` | FunctionDef | См. реализацию | [L31](../../../../tests/test_skin_neural_reference.py#L31) |
| `test_capacity_initial_equivalence_and_active_learning_paths` | FunctionDef | См. реализацию | [L44](../../../../tests/test_skin_neural_reference.py#L44) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_weighted_solver_matches_augmented_lstsq_and_excludes_people` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_weighted_solver_matches_augmented_lstsq_and_excludes_people():
    from skin_neural_reference import reference_correction
    rng=np.random.default_rng(218)
    bank=rng.normal(size=(13,4));query=rng.normal(size=(2,4));residual=rng.normal(size=(13,3))
    mask=np.ones((2,13),bool);mask[0,:4]=False;mask[1,4:8]=False
    got=reference_correction(*[torch.tensor(v) for v in (query,bank,residual,mask)],'affine').numpy()
    expected=[]
    for q,keep in zip(query,mask):
        b=bank[keep];y=residual[keep];logs=-.5*np.mean((q-b)**2,axis=1)
        w=np.exp(logs-logs.max());w/=w.sum();xm=w@b;ym=w@y
        design=np.vstack([(b-xm)*np.sqrt(w[:,None]),.1*np.eye(4)])
        target=np.vstack([(y-ym)*np.sqrt(w[:,None]),np.zeros((4,3))])
        coef=np.linalg.lstsq(design,target,rcond=None)[0];expected.append(ym+(q-xm)@coef)
    np.testing.assert_allclose(got,expected,atol=1e-12,rtol=1e-11)
    altered=residual.copy();altered[:4]+=1e6
    result=reference_correction(torch.tensor(query[:1]),torch.tensor(bank),torch.tensor(altered),torch.tensor(mask[:1]),'affine')
    np.testing.assert_array_equal(result.numpy()[0],got[0])
    with pytest.raises(ValueError,match='support'):
        reference_correction(torch.tensor(query),torch.tensor(bank),torch.tensor(residual),torch.zeros((2,13),dtype=torch.bool),'affine')
```

</details>

### `test_solver_gradients_and_reference_mean_constant_limit` · L31

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_solver_gradients_and_reference_mean_constant_limit():
    from skin_neural_reference import reference_correction
    torch.manual_seed(87)
    q=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    b=torch.randn(6,3,dtype=torch.float64,requires_grad=True)
    y=torch.randn(6,3,dtype=torch.float64,requires_grad=True)
    keep=torch.ones(2,6,dtype=torch.bool);keep[0,:2]=False
    assert torch.autograd.gradcheck(lambda a,c,d:reference_correction(a,c,d,keep,'affine'),(q,b,y),eps=1e-6,atol=1e-5,rtol=1e-4)
    constant=torch.tensor([2.,-3.,4.],dtype=torch.float64).expand(6,-1)
    for kind in ('affine','mean'):
        torch.testing.assert_close(reference_correction(q,b,constant,keep,kind),constant[:2])
```

</details>

### `test_capacity_initial_equivalence_and_active_learning_paths` · L44

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_capacity_initial_equivalence_and_active_learning_paths():
    from skin_neural_reference import ColorAdapter,core_features
    from skin_capture_model import CaptureColor
    torch.manual_seed(41);core=CaptureColor('mixture');core.eval();tokens=torch.randn(5,64,18)
    with torch.no_grad():
        p,c=core_features(core,tokens);expected=core(tokens)[0]
    torch.testing.assert_close(p,expected,rtol=0,atol=0)
    assert c.shape==(5,512) and sum(v.numel() for v in core.parameters())==929297
    x=torch.randn(5,551);b=torch.randn(9,551);y=torch.randn(9,3);keep=torch.ones(5,9,dtype=torch.bool)
    initial=[]
    for arm in ('residual','mean','affine'):
        torch.manual_seed(9);head=ColorAdapter(arm)
        assert sum(v.numel() for v in head.parameters())==193795
        assert 929297+sum(v.numel() for v in head.parameters())<=1129297
        initial.append(head(x,b,y,keep));assert torch.equal(initial[-1],torch.zeros(5,3))
        with torch.no_grad():head.output.weight.fill_(.03)
        head(x,b,y,keep).square().mean().backward()
        for parameter in head.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert head.features[0].weight.grad.abs().sum()>0
```

</details>
