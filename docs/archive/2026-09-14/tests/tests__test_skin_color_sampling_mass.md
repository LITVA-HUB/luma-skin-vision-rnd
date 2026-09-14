# `tests/test_skin_color_sampling_mass.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_color_sampling_mass.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `df2218e0aa2e9e1e35521df1dca7bd621e9253e48d13e0a40c234b88ef3bd261`. Строк: **20**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_color_sampling import sampling_distribution as original
from skin_color_sampling_mass import sampling_distribution
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_person_mass_preserved_when_color_association_removed` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_color_sampling_mass.py#L9) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_person_mass_preserved_when_color_association_removed` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_person_mass_preserved_when_color_association_removed():
    d={'site':np.array(['a','a','b','c','d']), 'patient':np.array(['p','p','p','q','q']),
       'target':np.array([[40,10,15],[40,10,15],[41,11,15],[42,9,16],[80,4,10]],float)}
    base,_,info=original(d,'color')
    for arm in ('person_mass','within_person_shuffle'):
        q,w,meta=sampling_distribution(d,arm,17)
        assert np.all(q>0) and abs(q.sum()-1)<1e-14
        for p in ('p','q'):np.testing.assert_allclose(q[d['patient']==p].sum(),base[d['patient']==p].sum(),rtol=1e-14)
        if arm=='within_person_shuffle':np.testing.assert_allclose(np.sort(meta['site_probability']),np.sort(info['site_probability']),rtol=1e-14)
        else:
            np.testing.assert_allclose(q[d['site']=='a'].sum(),q[d['site']=='b'].sum(),rtol=1e-14)
            np.testing.assert_allclose(q[d['site']=='c'].sum(),q[d['site']=='d'].sum(),rtol=1e-14)
```

</details>
