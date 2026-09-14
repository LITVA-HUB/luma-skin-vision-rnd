# `tests/test_skin_loss_field.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_loss_field.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `56f892d9789094e6d6a552708001d51dd7121a799aca89cb8a80feaf02902c50`. Строк: **34**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import torch
from skin_loss_field import risk_embedding, affine_weights, LossFieldImage
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_risk_embedding_preserves_full_candidate_loss` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_loss_field.py#L9) |
| `test_affine_weights_remove_positivity_but_preserve_unit_sum` | FunctionDef | См. реализацию | [L19](../../../../tests/test_skin_loss_field.py#L19) |
| `test_shared_backbone_has_finite_gradients_for_field_heads` | FunctionDef | См. реализацию | [L27](../../../../tests/test_skin_loss_field.py#L27) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_risk_embedding_preserves_full_candidate_loss` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_risk_embedding_preserves_full_candidate_loss():
    rng=np.random.default_rng(31);cost=rng.random((8,37))*20
    phi,center,scale=risk_embedding(cost)
    weights=rng.dirichlet(np.ones(8),size=5);truth=np.eye(8)[[0,2,4,6,7]]
    expected=np.mean(((weights-truth)@cost)**2,axis=1)/(scale**2)
    actual=np.sum(((weights-truth)@phi)**2,axis=1)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(center,cost.mean(0))
```

</details>

### `test_affine_weights_remove_positivity_but_preserve_unit_sum` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_affine_weights_remove_positivity_but_preserve_unit_sum():
    logits=torch.tensor([[1.,-3.,2.],[10.,-8.,0.]])
    w=affine_weights(logits)
    torch.testing.assert_close(w.sum(1),torch.ones(2))
    assert (w<0).any()
    torch.testing.assert_close(affine_weights(logits+5),w)
```

</details>

### `test_shared_backbone_has_finite_gradients_for_field_heads` · L27

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_shared_backbone_has_finite_gradients_for_field_heads():
    torch.manual_seed(17);model=LossFieldImage(12)
    p,g,h,logits=model(torch.rand(2,64,18))
    assert p.shape==(2,3) and h.shape==(2,4,3) and logits.shape==(2,12)
    loss=logits.softmax(1).square().sum()+p.square().mean()+g.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert 900_000<sum(p.numel() for p in model.parameters())<1_100_000
```

</details>
