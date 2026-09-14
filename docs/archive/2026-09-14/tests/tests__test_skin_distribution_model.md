# `tests/test_skin_distribution_model.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_distribution_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `e56b07a8c203e141e3f3c2e3be30d7a68637f5e574be882c632b4b3e3a304496`. Строк: **58**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
from skin_distribution_model import ColorDistribution, density_nll, quadrature, color_decision
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_density_matches_torch_distribution_and_has_finite_gradients` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_distribution_model.py#L9) |
| `test_quadrature_preserves_gaussian_mixture_mean_and_covariance` | FunctionDef | См. реализацию | [L24](../../../../tests/test_skin_distribution_model.py#L24) |
| `test_decision_minimizes_enumerated_risk_without_target` | FunctionDef | См. реализацию | [L39](../../../../tests/test_skin_distribution_model.py#L39) |
| `test_shared_model_and_gaussian_collapse` | FunctionDef | См. реализацию | [L50](../../../../tests/test_skin_distribution_model.py#L50) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_density_matches_torch_distribution_and_has_finite_gradients` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_density_matches_torch_distribution_and_has_finite_gradients():
    torch.manual_seed(42)
    means=torch.randn(7,4,3,dtype=torch.float64,requires_grad=True)
    scales=torch.rand(7,4,3,dtype=torch.float64,requires_grad=True)+.1
    logits=torch.randn(7,4,dtype=torch.float64,requires_grad=True)
    y=torch.randn(7,3,dtype=torch.float64)
    expected=-torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Independent(torch.distributions.Normal(means,scales),1)).log_prob(y)
    actual=density_nll(y,means,scales,logits)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    actual.mean().backward()
    assert torch.isfinite(means.grad).all() and torch.isfinite(logits.grad).all()
```

</details>

### `test_quadrature_preserves_gaussian_mixture_mean_and_covariance` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_quadrature_preserves_gaussian_mixture_mean_and_covariance():
    m=np.array([[[52,14,18],[66,9,21]]],float)
    s=np.array([[[2,1,3],[1,2,1]]],float);p=np.array([[.3,.7]])
    for order in (2,3):
        nodes,w=quadrature(m,s,p,order)
        mean=(m*p[...,None]).sum(1)
        np.testing.assert_allclose(w.sum(1),1,atol=1e-14)
        np.testing.assert_allclose((nodes*w[...,None]).sum(1),mean,atol=1e-12)
        for i in range(3):
            for j in range(3):
                cov=((nodes[:,:,i]-mean[:,i,None])*(nodes[:,:,j]-mean[:,j,None])*w).sum(1)
                expected=(((m[:,:,i]-mean[:,i,None])*(m[:,:,j]-mean[:,j,None])+(s[:,:,i]**2 if i==j else 0))*p).sum(1)
                np.testing.assert_allclose(cov,expected,atol=1e-11)
```

</details>

### `test_decision_minimizes_enumerated_risk_without_target` · L39

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_decision_minimizes_enumerated_risk_without_target():
    m=np.array([[[40,12,20],[70,12,20]]],float)
    s=np.full_like(m,.05);p=np.array([[.8,.2]])
    out=color_decision(m,s,p,3)
    assert out['expected_error'][0] <= out['mean_expected_error'][0]
    assert abs(out['prediction'][0,0]-40)<1
    assert out['mean'][0,0]==46
    equal=color_decision(m[:,0:1],s[:,0:1],np.ones((1,1)),3)
    assert np.linalg.norm(equal['prediction']-m[:,0])<.1
```

</details>

### `test_shared_model_and_gaussian_collapse` · L50

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_shared_model_and_gaussian_collapse():
    torch.manual_seed(17);model=ColorDistribution()
    x=torch.rand(2,64,18)
    point,logits,means,scales=model(x)
    assert means.shape==scales.shape==(2,4,3)
    assert torch.all(scales>=.05)
    torch.testing.assert_close(scales,torch.ones_like(scales))
    torch.testing.assert_close(point,(means*logits.softmax(1)[...,None]).sum(1))
    assert sum(p.numel() for p in model.parameters())<1_000_000
```

</details>
