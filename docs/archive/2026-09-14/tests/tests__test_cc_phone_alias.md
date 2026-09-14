# `tests/test_cc_phone_alias.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_phone_alias.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `a150b52483976d6d4d2a1748492e13a8f5a46593ec9b2199fa30383cc91283e7`. Строк: **31**.

## Зависимости

```python
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
from cc_phone_alias import open_camera_rgb
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_single_rgb_key_is_read_without_transform` | FunctionDef | См. реализацию | [L13](../../../../tests/test_cc_phone_alias.py#L13) |
| `test_ambiguous_or_multispectral_input_is_rejected` | FunctionDef | См. реализацию | [L23](../../../../tests/test_cc_phone_alias.py#L23) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_single_rgb_key_is_read_without_transform` · L13

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('key', ['samsung', 'MIS'])
def test_single_rgb_key_is_read_without_transform(tmp_path,key):
    x = np.arange(18,dtype=np.float32).reshape(2,3,3)
    path = tmp_path/"samsung.h5"
    with h5py.File(path,"w") as f:
        f[key] = x
    with open_camera_rgb(path,"samsung") as actual:
        np.testing.assert_array_equal(actual[:],x)
```

</details>

### `test_ambiguous_or_multispectral_input_is_rejected` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('second,channels', [(True, 3), (False, 16)])
def test_ambiguous_or_multispectral_input_is_rejected(tmp_path,second,channels):
    path = tmp_path/"samsung.h5"
    with h5py.File(path,"w") as f:
        f["MIS"] = np.ones((2,3,channels),dtype=np.float32)
        if second:
            f["samsung"] = np.ones((2,3,3),dtype=np.float32)
    with pytest.raises(ValueError):
        with open_camera_rgb(path,"samsung"):
            pass
```

</details>
