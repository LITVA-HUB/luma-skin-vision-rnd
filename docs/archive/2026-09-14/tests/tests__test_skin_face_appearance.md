# `tests/test_skin_face_appearance.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_face_appearance.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `ca04b81d346fe78fb8407c531c24fe22518bbd2d6bedd1cd7cac150050efe63a`. Строк: **28**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_color_reference_depends_only_on_masked_pixels_and_empty_is_missing` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_face_appearance.py#L9) |
| `test_surrogate_color_error_is_zero_for_matching_masks` | FunctionDef | См. реализацию | [L22](../../../../tests/test_skin_face_appearance.py#L22) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_color_reference_depends_only_on_masked_pixels_and_empty_is_missing` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_color_reference_depends_only_on_masked_pixels_and_empty_is_missing():
    from skin_face_appearance import appearance
    rgb = np.zeros((8,8,3),np.uint8)
    rgb[:4] = [180,120,100]
    mask = np.zeros((8,8),bool)
    mask[:4] = True
    a = appearance(rgb,mask)
    rgb[4:] = [10,250,70]
    np.testing.assert_array_equal(a['median_lab'],appearance(rgb,mask)['median_lab'])
    np.testing.assert_array_equal(a['mean_lab'],appearance(rgb,mask)['mean_lab'])
    assert appearance(rgb,np.zeros_like(mask)) is None
```

</details>

### `test_surrogate_color_error_is_zero_for_matching_masks` · L22

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_surrogate_color_error_is_zero_for_matching_masks():
    from skin_face_appearance import appearance, color_error
    rng = np.random.default_rng(123)
    rgb = rng.integers(0,256,(10,10,3),dtype=np.uint8)
    a = appearance(rgb,np.ones((10,10),bool))
    assert color_error(a,a)['median_delta_e00'] == 0
    assert color_error(a,a)['mean_delta_e00'] == 0
```

</details>
