# `tests/test_skin_correction_transfer.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_correction_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Camera-stable whole-person folds and four-core inclusion/exclusion routing.

SHA-256 исходника: `c24a17672597b395f169feba198bf7809fc5c5b92b376632dbe150819a20c4da`. Строк: **45**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_four_folds_match_between_mixed_and_single_camera_banks` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_correction_transfer.py#L9) |
| `test_four_core_routing_distinguishes_person_exclusion_from_inclusion` | FunctionDef | См. реализацию | [L25](../../../../tests/test_skin_correction_transfer.py#L25) |
| `test_prediction_batches_keep_each_domain_separate_and_restore_row_order` | FunctionDef | См. реализацию | [L35](../../../../tests/test_skin_correction_transfer.py#L35) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_four_folds_match_between_mixed_and_single_camera_banks` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_four_folds_match_between_mixed_and_single_camera_banks():
    from skin_correction_transfer import four_folds
    person=np.repeat(np.arange(24),2);camera=np.where(person<8,'SLR','ipod')
    f=four_folds(person,camera)
    for device in ('SLR','ipod'):
        ix=camera==device
        np.testing.assert_array_equal(f[ix],four_folds(person[ix],camera[ix]))
    for fold in range(4):
        a=f!=fold;b=~a
        assert len(set(person[b]))==6 and len(set(person[a]))==18
        assert set(person[a]).isdisjoint(person[b])
        assert len(set(person[b&(camera=='SLR')]))==2
        assert len(set(person[b&(camera=='ipod')]))==4
    with pytest.raises(ValueError):four_folds(person[:-2],camera[:-2])
```

</details>

### `test_four_core_routing_distinguishes_person_exclusion_from_inclusion` · L25

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_four_core_routing_distinguishes_person_exclusion_from_inclusion():
    from skin_correction_transfer import route_tables
    f=np.array([0,1,2,3,2,1]);values=np.stack([np.full((len(f),3),k,dtype=float) for k in range(4)])
    oof,seen=route_tables(values,f)
    np.testing.assert_array_equal(oof[:,0],f);np.testing.assert_array_equal(seen[:,0],(f+1)%4)
    changed=values.copy()
    for i,k in enumerate(f):changed[np.arange(4)!=k,i]+=1000
    np.testing.assert_array_equal(route_tables(changed,f)[0],oof)
```

</details>

### `test_prediction_batches_keep_each_domain_separate_and_restore_row_order` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_prediction_batches_keep_each_domain_separate_and_restore_row_order():
    from skin_correction_transfer import predict_domains
    domains={'known':np.array([1,0,1,0,1],bool),'unseen':np.array([0,1,0,1,0],bool)}
    calls=[]
    def callback(indices):
        calls.append(indices.copy())
        return np.repeat((indices+len(indices)/100)[:,None],3,axis=1).astype(np.float32)
    p=predict_domains(5,domains,callback)
    np.testing.assert_array_equal(calls[0],[0,2,4]);np.testing.assert_array_equal(calls[1],[1,3])
    np.testing.assert_allclose(p[:,0],[.03,1.02,2.03,3.02,4.03],atol=1e-6)
    with pytest.raises(ValueError):predict_domains(5,{'a':np.ones(5,bool),'b':np.ones(5,bool)},callback)
```

</details>
