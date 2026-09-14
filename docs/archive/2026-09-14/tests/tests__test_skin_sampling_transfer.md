# `tests/test_skin_sampling_transfer.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_sampling_transfer.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `bbc38926a082cbc0bb4dbfe63d74b04a429470d138d5d417084925f37a9e697d`. Строк: **30**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_color_sampling import sampling_distribution as original
from skin_sampling_transfer import sampling_distribution,protocols
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_combined_distribution_equalizes_people_preserving_color_ratios` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_sampling_transfer.py#L9) |
| `test_camera_held_out_evaluation_is_separate_from_known_camera` | FunctionDef | См. реализацию | [L22](../../../../tests/test_skin_sampling_transfer.py#L22) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_combined_distribution_equalizes_people_preserving_color_ratios` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_combined_distribution_equalizes_people_preserving_color_ratios():
    d={'site':np.array(['a','a','b','c','d']), 'patient':np.array(['p','p','p','q','q']),
       'target':np.array([[40,10,15],[40,10,15],[41,11,15],[42,9,16],[80,4,10]],float)}
    base,_,_=original(d,'color');q,_,_=sampling_distribution(d,'person_color')
    for person in ('p','q'):
        ix=d['patient']==person
        np.testing.assert_allclose(q[ix].sum(),.5,rtol=1e-14)
        np.testing.assert_allclose(q[ix]/q[ix].sum(),base[ix]/base[ix].sum(),rtol=1e-14)
    for arm in ('image','person_site','site','color','color_ipw'):
        a=original(d,arm);b=sampling_distribution(d,arm)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
```

</details>

### `test_camera_held_out_evaluation_is_separate_from_known_camera` · L22

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_camera_held_out_evaluation_is_separate_from_known_camera():
    tr={'device':np.array(['SLR','SLR','ipod']), 'patient':np.array(['a','b','c'])}
    va={'device':np.array(['SLR','ipod']), 'patient':np.array(['d','e'])}
    pp=list(protocols(tr,va))
    assert len(pp)==3 and set(pp[0][2])=={'known'}
    for name,t,e in pp[1:]:
        assert set(t['device'])==set(e['known']['device'])
        assert set(t['device']).isdisjoint(e['unseen']['device'])
        assert set(t['patient']).isdisjoint(set(e['known']['patient'])|set(e['unseen']['patient']))
```

</details>
