# `tests/test_skin_lapa_prepare.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_lapa_prepare.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `3da83f0eb31ef0d87ada7e76e47be3cd79e257e5a91c9f76c7932b5e36800ea4`. Строк: **39**.

## Зависимости

```python
import sys
from pathlib import Path
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_lapa_archive_paths_are_bounded` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_lapa_prepare.py#L9) |
| `records` | FunctionDef | См. реализацию | [L17](../../../../tests/test_skin_lapa_prepare.py#L17) |
| `test_lapa_preserves_official_test_and_excludes_source_group_or_byte_duplicates` | FunctionDef | См. реализацию | [L23](../../../../tests/test_skin_lapa_prepare.py#L23) |
| `test_lapa_missing_or_duplicate_pairs_are_rejected` | FunctionDef | См. реализацию | [L33](../../../../tests/test_skin_lapa_prepare.py#L33) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_lapa_archive_paths_are_bounded` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_lapa_archive_paths_are_bounded(tmp_path):
    from skin_lapa_prepare import member_target
    assert member_target(tmp_path, 'LaPa/train/images/123_0.jpg').is_relative_to(tmp_path)
    for name in ['../outside', 'LaPa/train/images/../../bad.jpg', '/abs.jpg', 'LaPa/train/images/C:x.jpg']:
        with pytest.raises(ValueError, match='path'):
            member_target(tmp_path, name)
```

</details>

### `test_lapa_preserves_official_test_and_excludes_source_group_or_byte_duplicates` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_lapa_preserves_official_test_and_excludes_source_group_or_byte_duplicates():
    from skin_lapa_prepare import select_rows
    source = (records('test', '123_0', 'a') + records('train', '123_1', 'b') +
              records('val', '500_0', 'a') + records('val', '700_0', 'c') +
              records('train', '800_0', 'c') + records('train', '900_0', 'd'))
    kept, dropped = select_rows(source)
    assert [(r['split'], r['stem']) for r in kept] == [('test','123_0'),('train','900_0'),('val','700_0')]
    assert len(dropped) == 3
```

</details>

### `test_lapa_missing_or_duplicate_pairs_are_rejected` · L33

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_lapa_missing_or_duplicate_pairs_are_rejected():
    from skin_lapa_prepare import select_rows
    source = records('train', '1_0', 'a')
    with pytest.raises(ValueError, match='triplet'):
        select_rows(source[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        select_rows(source + source[:1])
```

</details>
