# `tests/test_fourier_ridge.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_fourier_ridge.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `941c762c8eae7b31cbf3250625a2fba68bd2c0d365a2d3c078d266055c472fa6`. Строк: **50**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from cc_fourier_ridge import fit_filter, predict_score
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_fourier_solution_matches_real_linear_system` | FunctionDef | См. реализацию | [L10](../../../../tests/test_fourier_ridge.py#L10) |
| `test_zero_ridge_recovers_known_circular_operator` | FunctionDef | См. реализацию | [L25](../../../../tests/test_fourier_ridge.py#L25) |
| `test_bias_ablation_removes_prior_coefficient_exactly` | FunctionDef | См. реализацию | [L35](../../../../tests/test_fourier_ridge.py#L35) |
| `test_bias_on_is_exact_original_fit` | FunctionDef | См. реализацию | [L45](../../../../tests/test_fourier_ridge.py#L45) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_fourier_solution_matches_real_linear_system` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fourier_solution_matches_real_linear_system():
    rng = np.random.default_rng(8)
    hist = rng.normal(size=(12,2,4,4))
    target = rng.normal(size=(12,4,4))
    weight = fit_filter(hist,target,.1)
    h = np.fft.fft2(hist).transpose(0,2,3,1)
    a = np.concatenate([h,np.ones((*h.shape[:-1],1))],axis=-1)
    y = np.fft.fft2(target)
    residual = np.einsum("nhwc,hwc->nhw",a,weight)-y
    gradient = np.einsum("nhwc,nhw->hwc",a.conj(),residual)/len(hist)+.1*weight
    np.testing.assert_allclose(gradient,0,atol=1e-12)
    score = predict_score(hist,weight)
    assert score.shape == target.shape and np.isfinite(score).all()
```

</details>

### `test_zero_ridge_recovers_known_circular_operator` · L25

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_zero_ridge_recovers_known_circular_operator():
    rng = np.random.default_rng(3)
    hist = rng.normal(size=(10,2,4,4))
    true = np.fft.fft2(rng.normal(size=(3,4,4))).transpose(1,2,0)
    target = predict_score(hist,true)
    fitted = fit_filter(hist,target,1e-12)
    unseen = rng.normal(size=(3,2,4,4))
    np.testing.assert_allclose(predict_score(unseen,fitted),predict_score(unseen,true),atol=1e-9)
```

</details>

### `test_bias_ablation_removes_prior_coefficient_exactly` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bias_ablation_removes_prior_coefficient_exactly():
    from cc_fourier_ridge_v2 import fit_filter as fit_v2
    rng=np.random.default_rng(14)
    hist=rng.normal(size=(10,2,4,4))
    target=rng.normal(size=(10,4,4))
    weight=fit_v2(hist,target,.01,bias=False)
    np.testing.assert_array_equal(weight[...,2],0)
    assert np.isfinite(predict_score(hist,weight)).all()
```

</details>

### `test_bias_on_is_exact_original_fit` · L45

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bias_on_is_exact_original_fit():
    from cc_fourier_ridge_v2 import fit_filter as fit_v2
    rng=np.random.default_rng(15)
    hist=rng.normal(size=(10,2,4,4))
    target=rng.normal(size=(10,4,4))
    np.testing.assert_array_equal(fit_filter(hist,target,.01),fit_v2(hist,target,.01,bias=True))
```

</details>
