# `tests/test_skin_issa_material.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_issa_material.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `2ce47f81fe197b81e7e97f5f07b67344e28a597b852d4422caf701fa408a98b4`. Строк: **19**.

## Зависимости

```python
import numpy as np
from scripts.skin_issa_material import fit_basis, encode, decode, subject_weights
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_subject_weighting_resists_record_multiplicity` | FunctionDef | См. реализацию | [L5](../../../../tests/test_skin_issa_material.py#L5) |
| `test_physical_transforms_roundtrip_without_training_on_evaluation_data` | FunctionDef | См. реализацию | [L16](../../../../tests/test_skin_issa_material.py#L16) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_subject_weighting_resists_record_multiplicity` · L5

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_subject_weighting_resists_record_multiplicity():
    x=np.array([[1.,0.,2.],[2.,1.,3.],[4.,3.,1.],[5.,1.,2.]])
    people=np.array(['a','a','b','c'])
    mean, basis=fit_basis(x,subject_weights(people))
    more=np.concatenate([x,x[:2]])
    m2,b2=fit_basis(more,subject_weights(np.concatenate([people,people[:2]])))
    np.testing.assert_allclose(m2,mean,atol=1e-12)
    np.testing.assert_allclose(basis@basis.T,b2@b2.T,atol=1e-12)
    np.testing.assert_allclose((x-mean)@basis@basis.T+mean,x,atol=1e-12)
```

</details>

### `test_physical_transforms_roundtrip_without_training_on_evaluation_data` · L16

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_physical_transforms_roundtrip_without_training_on_evaluation_data():
    x=np.array([[.01,.3,.8],[.6,.9,.99]])
    for method in ('reflectance','density','logit'):
        np.testing.assert_allclose(decode(encode(x,method),method),x,atol=1e-14)
```

</details>
