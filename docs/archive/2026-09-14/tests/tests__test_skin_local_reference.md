# `tests/test_skin_local_reference.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_local_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Local support controls: uniform limit, stable mass, held-target exclusion.

SHA-256 исходника: `b4711cf1770637e6a96a4841638b51c998a7a9e5a28734c6e8cc9f388c2ab6a8`. Строк: **37**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_uniform_local_affine_is_global_ridge_with_unpenalized_intercept` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_local_reference.py#L8) |
| `test_kernel_mass_survives_extreme_distances_and_is_scale_invariant` | FunctionDef | См. реализацию | [L19](../../../../tests/test_skin_local_reference.py#L19) |
| `test_predictions_do_not_require_query_color_or_identifiers` | FunctionDef | См. реализацию | [L28](../../../../tests/test_skin_local_reference.py#L28) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_uniform_local_affine_is_global_ridge_with_unpenalized_intercept` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_uniform_local_affine_is_global_ridge_with_unpenalized_intercept():
    from skin_local_reference import local_affine
    rng=np.random.default_rng(199);x=rng.normal(size=(17,4));y=rng.normal(size=(17,3));q=rng.normal(size=4)
    xc=x-x.mean(0);yc=y-y.mean(0)
    coef=np.linalg.solve(xc.T@xc+np.eye(4),xc.T@yc)
    expected=y.mean(0)+(q-x.mean(0))@coef
    np.testing.assert_allclose(local_affine(x,y,q,np.ones(17)),expected,atol=1e-14,rtol=1e-12)
    shift=np.array([40.,-5.,20.])
    np.testing.assert_allclose(local_affine(x,y+shift,q,np.ones(17)),expected+shift,atol=1e-13,rtol=1e-12)
```

</details>

### `test_kernel_mass_survives_extreme_distances_and_is_scale_invariant` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_kernel_mass_survives_extreme_distances_and_is_scale_invariant():
    from skin_local_reference import normalized_weights
    logs=np.array([-1000001.,-1000000.,-1001000.])
    w=normalized_weights(logs)
    assert np.isfinite(w).all() and (w>=0).all()
    np.testing.assert_allclose(w.sum(),3,atol=1e-15)
    np.testing.assert_allclose(w,normalized_weights(logs+1000000),atol=1e-14,rtol=1e-12)
```

</details>

### `test_predictions_do_not_require_query_color_or_identifiers` · L28

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_predictions_do_not_require_query_color_or_identifiers():
    from skin_local_reference import predict_bank
    rng=np.random.default_rng(62);x=rng.normal(size=(15,4));y=rng.normal(size=(15,3))+np.array([50,10,15]);q=rng.normal(size=(3,4))
    got=predict_bank(x,y,q)
    assert set(got['predictions'])=={'global_ridge','global_mean','appearance_mean','appearance_affine','color_mean','color_affine'}
    for v in got['predictions'].values():assert v.shape==(3,3) and np.isfinite(v).all()
    reverse=predict_bank(x[::-1],y[::-1],q)
    for name,v in got['predictions'].items():np.testing.assert_allclose(v,reverse['predictions'][name],atol=1e-12,rtol=1e-11)
    assert got['weights_appearance'].shape==(3,15)
    np.testing.assert_allclose(got['weights_color'].sum(1),15,atol=1e-13)
```

</details>
