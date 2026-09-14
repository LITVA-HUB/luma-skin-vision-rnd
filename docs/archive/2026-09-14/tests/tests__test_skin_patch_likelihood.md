# `tests/test_skin_patch_likelihood.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_patch_likelihood.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `130f1adfca80d70ae2301ed21ab363253b8b2f90e5d465f4dbba21d343282444`. Строк: **26**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_patch_likelihood import bag_loglik,component_update
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_single_gaussian_patch_scatter_cancels_across_color_candidates` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_patch_likelihood.py#L8) |
| `test_non_gaussian_bag_contains_information_absent_from_mean` | FunctionDef | См. реализацию | [L14](../../../../tests/test_skin_patch_likelihood.py#L14) |
| `test_one_component_mean_fit_matches_image_weighted_ridge` | FunctionDef | См. реализацию | [L21](../../../../tests/test_skin_patch_likelihood.py#L21) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_single_gaussian_patch_scatter_cancels_across_color_candidates` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_single_gaussian_patch_scatter_cancels_across_color_candidates():
    rng=np.random.default_rng(17);x=rng.normal(size=(3,8,3));means=rng.normal(size=(5,1,3))
    a=bag_loglik(x,means,np.eye(3)[None],np.ones(1));b=bag_loglik(x.mean(1,keepdims=True),means,np.eye(3)[None],np.ones(1))
    np.testing.assert_allclose(a-a[:,:1],b-b[:,:1],atol=1e-12)
```

</details>

### `test_non_gaussian_bag_contains_information_absent_from_mean` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_non_gaussian_bag_contains_information_absent_from_mean():
    x=np.array([[[-1.],[1.]]]);means=np.array([[[-1.],[1.]],[[0.],[0.]]]);cov=np.array([[[.05]],[[.05]]]);pi=np.ones(2)/2
    a=bag_loglik(x,means,cov,pi);b=bag_loglik(x.mean(1,keepdims=True),means,cov,pi)
    assert a.argmax(1)[0]==0 and b.argmax(1)[0]==1
    np.testing.assert_allclose(a,bag_loglik(x[:,::-1],means,cov,pi),atol=1e-12)
```

</details>

### `test_one_component_mean_fit_matches_image_weighted_ridge` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_one_component_mean_fit_matches_image_weighted_ridge():
    rng=np.random.default_rng(8);d=np.column_stack([np.ones(10),rng.normal(size=(10,3))]);x=rng.normal(size=(10,7,3))
    w,_,_=component_update(d,x,np.ones((10,7,1)),1.)
    penalty=np.eye(4);penalty[0,0]=0
    expected=np.linalg.solve(d.T@d+penalty,d.T@x.mean(1))
    np.testing.assert_allclose(w[0],expected,rtol=1e-12,atol=1e-12)
```

</details>
