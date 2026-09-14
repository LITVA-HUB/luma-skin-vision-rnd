# `tests/test_cc_phone_data.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_phone_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Sparse public-phone acquisition mechanics; no benchmark claims.

SHA-256 исходника: `d7e8faee3228d5866f8171b55e7496555f8a285716fc29cfd4ea625cd32a6156`. Строк: **39**.

## Зависимости

```python
import importlib.util
import sys
from pathlib import Path
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `module` | FunctionDef | См. реализацию | [L11](../../../../tests/test_cc_phone_data.py#L11) |
| `test_split_archive_ranges_cover_cross_part_record_without_extra_bytes` | FunctionDef | См. реализацию | [L20](../../../../tests/test_cc_phone_data.py#L20) |
| `test_scene_completeness_requires_two_phone_targets_and_keeps_privacy_masks` | FunctionDef | См. реализацию | [L30](../../../../tests/test_cc_phone_data.py#L30) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_split_archive_ranges_cover_cross_part_record_without_extra_bytes` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_split_archive_ranges_cover_cross_part_record_without_extra_bytes():
    m = module()
    assert m.split_ranges([10, 20, 5], 8, 25) == [(0, 8, 9), (1, 0, 19), (2, 0, 2)]
    assert m.split_ranges([10, 20], 10, 20) == [(1, 0, 19)]
    with pytest.raises(ValueError):
        m.split_ranges([10, 20], 29, 2)
    with pytest.raises(ValueError):
        m.split_ranges([10, 20], -1, 1)
```

</details>

### `test_scene_completeness_requires_two_phone_targets_and_keeps_privacy_masks` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_scene_completeness_requires_two_phone_targets_and_keeps_privacy_masks():
    m = module()
    scene = 'outdoor/1'
    names = [f'beyond-unzip/beyondRGB/{scene}/{name}' for name in m.REQUIRED]
    rows = {name: {'path': name, 'compressed_bytes': 1} for name in names}
    mask = f'beyond-unzip/beyondRGB/{scene}/NT/samsung_blurred_areas_detection.json'
    rows[mask] = {'path': mask, 'compressed_bytes': 1}
    assert any(r['path'] == mask for r in m.scene_members(rows, scene))
    del rows[f'beyond-unzip/beyondRGB/{scene}/WT/oppo.h5']
    assert m.scene_members(rows, scene) is None
```

</details>
