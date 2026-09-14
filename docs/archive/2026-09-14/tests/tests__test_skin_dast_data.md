# `tests/test_skin_dast_data.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_dast_data.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `20fc6888f462b7165d5c6918757405019d7f269d2afd9f5c16865b453d425a23`. Строк: **63**.

## Зависимости

```python
import io
import sys
import zipfile
from pathlib import Path
import pytest
from PIL import Image
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `cmf` | FunctionDef | См. реализацию | [L12](../../../../tests/test_skin_dast_data.py#L12) |
| `archive` | FunctionDef | См. реализацию | [L16](../../../../tests/test_skin_dast_data.py#L16) |
| `test_dast_native_measurements_preserve_unknown_sites` | FunctionDef | См. реализацию | [L30](../../../../tests/test_skin_dast_data.py#L30) |
| `test_dast_filename_alias_is_unique_and_photos_remain_byte_exact` | FunctionDef | См. реализацию | [L43](../../../../tests/test_skin_dast_data.py#L43) |
| `test_dast_archive_rejects_unsafe_paths_before_writing` | FunctionDef | См. реализацию | [L58](../../../../tests/test_skin_dast_data.py#L58) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_dast_native_measurements_preserve_unknown_sites` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_dast_native_measurements_preserve_unknown_sites():
    from skin_dast_data import parse_cmf
    known = parse_cmf(cmf())[0]
    assert known['lab_native'] == [55., 10., 14.]
    assert known['site_code'] == 'S'
    assert known['illuminant'] is None
    assert parse_cmf(cmf('Measure 3'))[0]['anatomical_site'] is None
    with pytest.raises(ValueError, match='finite'):
        parse_cmf(cmf().replace('\t55\t', '\tnan\t'))
    with pytest.raises(ValueError, match='header'):
        parse_cmf('not a measurement table')
```

</details>

### `test_dast_filename_alias_is_unique_and_photos_remain_byte_exact` · L43

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_dast_filename_alias_is_unique_and_photos_remain_byte_exact(tmp_path):
    from skin_dast_data import ingest_dast
    raw = archive(tmp_path/'original.zip')
    result = ingest_dast(tmp_path/'original.zip', tmp_path/'out')
    assert result['subjects'] == 1 and result['photographs'] == 1
    row = result['images'][0]
    assert row['subject_id'] == '002'
    assert row['camera'] is None and row['exposure'] is None
    assert Path(row['path']).read_bytes() == raw
    assert row['role'] == 'external_diagnostic'
    assert row['ground_truth_lab'] is None
    assert row['filename_alias_used'] is True
```

</details>

### `test_dast_archive_rejects_unsafe_paths_before_writing` · L58

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('bad', ['../escape.jpg', '/abs.jpg', 'example-data/../x.jpg', 'C:/x.jpg'])
def test_dast_archive_rejects_unsafe_paths_before_writing(tmp_path, bad):
    from skin_dast_data import ingest_dast
    archive(tmp_path/'original.zip', bad)
    with pytest.raises(ValueError, match='path'):
        ingest_dast(tmp_path/'original.zip', tmp_path/'out')
    assert not (tmp_path/'out').exists()
```

</details>
