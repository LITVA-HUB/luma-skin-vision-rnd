# `tests/test_skin_color_hull_control.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_color_hull_control.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `32f4677e0f7ea0b19158827c63df58f7a3f8a485e30bf3ac63ef4a915432e467`. Строк: **18**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_color_hull_control import make_hull,project_hull
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_tetrahedron_projection_and_kkt` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_color_hull_control.py#L8) |
| `test_projection_leaves_observed_colors_fixed` | FunctionDef | См. реализацию | [L15](../../../../tests/test_skin_color_hull_control.py#L15) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_tetrahedron_projection_and_kkt` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_tetrahedron_projection_and_kkt():
    palette=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    p,receipt=project_hull(np.array([[1.,1.,1.],[-1.,0.,0.],[.1,.2,.3]]),make_hull(palette))
    np.testing.assert_allclose(p,[[1/3,1/3,1/3],[0,0,0],[.1,.2,.3]],atol=1e-8)
    assert receipt['max_feasibility_violation']<1e-8 and receipt['max_relative_stationarity']<1e-8
```

</details>

### `test_projection_leaves_observed_colors_fixed` · L15

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_projection_leaves_observed_colors_fixed():
    rng=np.random.default_rng(17);palette=rng.normal(size=(30,3))
    out,_=project_hull(palette,make_hull(palette))
    np.testing.assert_array_equal(out,palette)
```

</details>
