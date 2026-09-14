# `tests/test_skin_color_sampling.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_color_sampling.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `721879dff07c1b2a52e62897db2f92ae74af353ba70e2eef76f34e16575205d9`. Строк: **33**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_color_sampling import sampling_distribution,draw_indices
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `example` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_color_sampling.py#L8) |
| `test_site_and_person_mass_are_independent_of_photo_duplicates` | FunctionDef | См. реализацию | [L13](../../../../tests/test_skin_color_sampling.py#L13) |
| `test_importance_correction_recovers_site_objective_exactly` | FunctionDef | См. реализацию | [L21](../../../../tests/test_skin_color_sampling.py#L21) |
| `test_image_draws_match_existing_control_and_weights_do_not_change_draws` | FunctionDef | См. реализацию | [L29](../../../../tests/test_skin_color_sampling.py#L29) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_site_and_person_mass_are_independent_of_photo_duplicates` · L13

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_site_and_person_mass_are_independent_of_photo_duplicates():
    d=example();q,_,_=sampling_distribution(d,'site')
    np.testing.assert_allclose([q[d['site']==s].sum() for s in ('a','b','c')],[1/3]*3)
    q,_,_=sampling_distribution(d,'person_site')
    np.testing.assert_allclose([q[d['patient']==p].sum() for p in ('p','q')],[.5,.5])
    np.testing.assert_allclose([q[d['site']==s].sum() for s in ('a','b','c')],[.25,.25,.5])
```

</details>

### `test_importance_correction_recovers_site_objective_exactly` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_importance_correction_recovers_site_objective_exactly():
    d=example();site,_,_=sampling_distribution(d,'site');q,w,_=sampling_distribution(d,'color_ipw')
    np.testing.assert_allclose(q*w,site,rtol=1e-14)
    assert w.max()<=2 and q.min()>0 and abs(q.sum()-1)<1e-14
    assert q[d['site']=='c'].sum()>q[d['site']=='a'].sum()
    loss=np.array([1,2,3,8,4,7.]);assert abs(np.dot(q,w*loss)-np.dot(site,loss))<1e-14
```

</details>

### `test_image_draws_match_existing_control_and_weights_do_not_change_draws` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_image_draws_match_existing_control_and_weights_do_not_change_draws():
    d=example();q,_,_=sampling_distribution(d,'image')
    np.testing.assert_array_equal(draw_indices(q,'image',17,1),np.random.default_rng(17001).integers(0,6,(31,32),dtype=np.int64))
    a,_,_=sampling_distribution(d,'color');b,_,_=sampling_distribution(d,'color_ipw')
    np.testing.assert_array_equal(draw_indices(a,'color',29,2),draw_indices(b,'color_ipw',29,2))
```

</details>
