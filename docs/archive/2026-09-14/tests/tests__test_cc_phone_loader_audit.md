# `tests/test_cc_phone_loader_audit.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_phone_loader_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `82b2d5de631b31f13378709216776a6122f2b107cb04f633693157ed929abc02`. Строк: **57**.

## Зависимости

```python
import importlib.util
from pathlib import Path
import h5py
import numpy as np
import pytest
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../tests/test_cc_phone_loader_audit.py#L8)

```python
SPEC = importlib.util.spec_from_file_location(
    'phone_audit', Path(__file__).parents[1] / 'scripts/cc_phone_loader_audit.py')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_loader_roles_reject_test_before_opening_pixels` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc_phone_loader_audit.py#L14) |
| `test_native_rgb_preserved_without_second_whitelevel_or_gamma` | FunctionDef | См. реализацию | [L19](../../../../tests/test_cc_phone_loader_audit.py#L19) |
| `test_patch_xy_polygon_uses_interior_and_reports_saturation` | FunctionDef | См. реализацию | [L31](../../../../tests/test_cc_phone_loader_audit.py#L31) |
| `test_malformed_or_outside_patch_does_not_silently_clip` | FunctionDef | См. реализацию | [L43](../../../../tests/test_cc_phone_loader_audit.py#L43) |
| `test_radiometric_scan_reports_invalid_values_without_hiding_them` | FunctionDef | См. реализацию | [L51](../../../../tests/test_cc_phone_loader_audit.py#L51) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_loader_roles_reject_test_before_opening_pixels` · L14

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_loader_roles_reject_test_before_opening_pixels():
    with pytest.raises(ValueError, match='reserved'):
        audit.loader_scenes({'scenes': [{'scene': 'outdoor/1', 'role': 'reserved_test'}]}, ['outdoor/1'])
```

</details>

### `test_native_rgb_preserved_without_second_whitelevel_or_gamma` · L19

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_native_rgb_preserved_without_second_whitelevel_or_gamma(tmp_path):
    path = tmp_path/'samsung.h5'
    expected = np.full((8, 12, 3), [.1, .4, .2], np.float32)
    with h5py.File(path, 'w') as f:
        f.create_dataset('samsung', data=expected)
    with audit.open_camera_rgb(path, 'samsung') as data:
        np.testing.assert_array_equal(data[:], expected)
    with pytest.raises(ValueError, match='camera'):
        with audit.open_camera_rgb(path, 'oppo'):
            pass
```

</details>

### `test_patch_xy_polygon_uses_interior_and_reports_saturation` · L31

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_patch_xy_polygon_uses_interior_and_reports_saturation():
    data = np.zeros((30, 50, 3), np.float32)
    data[5:16, 25:41] = [.1, .3, .2]
    corners = [[25, 5], [40, 5], [40, 15], [25, 15]]
    stats = audit.patch_stats(data, corners, inset=.5)
    np.testing.assert_allclose(stats['median_rgb'], [.1, .3, .2])
    assert stats['saturated_pixel_fraction'] == 0
    assert stats['sample_count'] > 20
    data[5:16, 25:41, 1] = 1
    assert audit.patch_stats(data, corners)['saturated_pixel_fraction'] == 1
```

</details>

### `test_malformed_or_outside_patch_does_not_silently_clip` · L43

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_malformed_or_outside_patch_does_not_silently_clip():
    data = np.zeros((30, 50, 3))
    with pytest.raises(ValueError, match='bounds'):
        audit.patch_stats(data, [[-1, 0], [5, 0], [5, 5], [-1, 5]])
    with pytest.raises(ValueError, match='convex'):
        audit.patch_stats(data, [[1, 1], [10, 10], [1, 10], [10, 1]])
```

</details>

### `test_radiometric_scan_reports_invalid_values_without_hiding_them` · L51

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_radiometric_scan_reports_invalid_values_without_hiding_them():
    data = np.full((5, 8, 3), 100/255, np.float32)
    stats = audit.radiometric_sample(data, stride=1)
    assert stats['fraction_on_8bit_grid'] == 1
    data[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        audit.radiometric_sample(data, stride=1)
```

</details>
