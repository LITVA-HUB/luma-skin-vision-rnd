# `tests/test_skin_offset_diagnostic.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_offset_diagnostic.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `ad1c83a9917a04bd30f3f429614ba68369e7d022bfb757b721f49009c2341592`. Строк: **24**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_offset_diagnostic import excluded_person_offsets
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_own_reference_cannot_change_own_correction` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_offset_diagnostic.py#L8) |
| `test_photo_replication_does_not_reweight_sites_or_people` | FunctionDef | См. реализацию | [L18](../../../../tests/test_skin_offset_diagnostic.py#L18) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_own_reference_cannot_change_own_correction` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_own_reference_cannot_change_own_correction():
    patient=np.array(['a','a','b','c']);site=np.array(['x','x','y','z'])
    pred=np.zeros((4,3));target=np.array([[1,2,3],[1,2,3],[4,5,6],[8,9,10]],float)
    a=excluded_person_offsets(pred,target,patient,site)
    changed=target.copy();changed[patient=='a']+=1000
    b=excluded_person_offsets(pred,changed,patient,site)
    np.testing.assert_array_equal(a[patient=='a'],b[patient=='a'])
    np.testing.assert_allclose(a[patient=='a'],[[6,7,8],[6,7,8]])
```

</details>

### `test_photo_replication_does_not_reweight_sites_or_people` · L18

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_photo_replication_does_not_reweight_sites_or_people():
    patient=np.array(['a','b','b','c']);site=np.array(['x','y','z','w'])
    pred=np.zeros((4,3));target=np.array([[0,0,0],[2,2,2],[6,6,6],[10,10,10]],float)
    a=excluded_person_offsets(pred,target,patient,site)
    ix=np.array([0,1,1,1,2,3])
    b=excluded_person_offsets(pred[ix],target[ix],patient[ix],site[ix])
    np.testing.assert_allclose(a[0],[7,7,7]);np.testing.assert_array_equal(a[0],b[0])
```

</details>
