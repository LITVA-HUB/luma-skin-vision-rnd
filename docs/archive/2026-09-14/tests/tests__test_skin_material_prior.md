# `tests/test_skin_material_prior.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_material_prior.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `05826f9eba220f4d140f08c8f4300d73fd49e46b913e37614da5e0eed865dad6`. Строк: **22**.

## Зависимости

```python
import numpy as np
from scripts.skin_material_prior import integration_matrix, lab_from_xyz, material_value_jacobian
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_integration_preserves_flat_reflectance_and_linear_interpolation` | FunctionDef | См. реализацию | [L5](../../../../tests/test_skin_material_prior.py#L5) |
| `test_material_jacobian_agrees_with_finite_difference` | FunctionDef | См. реализацию | [L14](../../../../tests/test_skin_material_prior.py#L14) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_integration_preserves_flat_reflectance_and_linear_interpolation` · L5

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_integration_preserves_flat_reflectance_and_linear_interpolation():
    wave=np.arange(360.,831.);knots=np.arange(400.,701.,10.)
    cmf=np.ones((len(wave),3))*[.7,1.,.4];spd=np.ones(len(wave))
    matrix=integration_matrix(knots,wave,cmf,spd)
    white=matrix.sum(0)
    np.testing.assert_allclose(white,[.7,1.,.4],atol=1e-12)
    np.testing.assert_allclose(lab_from_xyz(np.full(31,.5)@matrix,white),[116*np.cbrt(.5)-16,0,0],atol=1e-12)
```

</details>

### `test_material_jacobian_agrees_with_finite_difference` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_material_jacobian_agrees_with_finite_difference():
    rng=np.random.default_rng(7);mu=rng.normal(size=31);basis=rng.normal(size=(31,8))*.1
    matrix=rng.uniform(.001,.1,size=(31,3));matrix/=matrix[:,1].sum();white=matrix.sum(0)
    value,jac=material_value_jacobian(mu,basis,matrix,white)
    def f(z):return lab_from_xyz((1/(1+np.exp(-(mu+basis@z))))@matrix,white)
    np.testing.assert_allclose(value,f(np.zeros(8)),atol=1e-12)
    for i in range(8):
        z=np.zeros(8);z[i]=1e-5
        np.testing.assert_allclose(jac[i],(f(z)-f(-z))/(2e-5),atol=2e-8)
```

</details>
