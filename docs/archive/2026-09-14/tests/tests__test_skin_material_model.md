# `tests/test_skin_material_model.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_material_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `f4d0c32a921da5b063afda5f0069e183b93c28b7c52044929f55577d9f11f2c0`. Строк: **34**.

## Зависимости

```python
import numpy as np
import torch
from scripts.skin_material_prior import material_value_jacobian
from scripts.skin_material_model import MaterialImage, ARMS
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `prior` | FunctionDef | См. реализацию | [L7](../../../../tests/test_skin_material_model.py#L7) |
| `test_all_arms_have_matched_states_and_finite_gradients` | FunctionDef | См. реализацию | [L14](../../../../tests/test_skin_material_model.py#L14) |
| `test_tangent_matches_material_value_and_derivative_at_origin` | FunctionDef | См. реализацию | [L26](../../../../tests/test_skin_material_model.py#L26) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_all_arms_have_matched_states_and_finite_gradients` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_all_arms_have_matched_states_and_finite_gradients():
    states=[]
    for arm in ARMS:
        torch.manual_seed(9);m=MaterialImage(arm,prior(),np.zeros(3),np.ones(3))
        states.append(m.state_dict())
        p,g,h,r=m(torch.ones(2,64,18)*.2)
        assert p.shape==(2,3) and h.shape==(2,4,3)
        (p.square().mean()+g.square().mean()).backward()
        assert all(torch.isfinite(q.grad).all() for q in m.parameters() if q.grad is not None)
    assert all(torch.equal(a,states[0][k]) for s in states[1:] for k,a in s.items())
```

</details>

### `test_tangent_matches_material_value_and_derivative_at_origin` · L26

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_tangent_matches_material_value_and_derivative_at_origin():
    p=prior();m=MaterialImage('material',p,np.zeros(3),np.ones(3)).double()
    z=torch.zeros(8,dtype=torch.double,requires_grad=True)
    value=m.material_decode(z)
    jac=torch.autograd.functional.jacobian(m.material_decode,z).T
    # Buffers originate as deployment float32; comparison uses their actual values.
    expected,j=material_value_jacobian(m.mu.numpy(),m.basis.numpy(),m.matrix.numpy(),m.white.numpy())
    np.testing.assert_allclose(value.detach().numpy(),expected,atol=1e-10)
    np.testing.assert_allclose(jac.detach().numpy(),j,atol=1e-10)
```

</details>
