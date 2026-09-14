# `tests/test_skin_uminho_train_expand_v2.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_uminho_train_expand_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `7592f713b7886ec3444cc12fce122f7b8e4aa3ab31bbd68d0159d1b4c055d6eb`. Строк: **27**.

## Зависимости

```python
import sys
from pathlib import Path
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_only_original_training_roles_and_deterministic_order` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_uminho_train_expand_v2.py#L9) |
| `test_source_paths_cannot_escape` | FunctionDef | См. реализацию | [L22](../../../../tests/test_skin_uminho_train_expand_v2.py#L22) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_only_original_training_roles_and_deterministic_order` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_only_original_training_roles_and_deterministic_order():
    from skin_uminho_train_expand_v2 import training_rows
    rows = [dict(name=f'{i}_reflectance.mat', role='train', size=i + 1) for i in range(19)]
    held = [dict(name='test.mat', role='test'), dict(name='val.mat', role='validation')]
    chosen = training_rows(list(reversed(rows)) + held)
    assert len(chosen) == 19
    assert all(r['role'] == 'train' for r in chosen)
    assert chosen == training_rows(rows + list(reversed(held)))
    with pytest.raises(ValueError, match='19'):
        training_rows(rows[:-1] + held)
```

</details>

### `test_source_paths_cannot_escape` · L22

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('name', ['../escape.mat', 'C:\\escape.mat', 'folder/escape.mat', 'folder\\escape.mat'])
def test_source_paths_cannot_escape(name):
    from skin_uminho_train_expand_v2 import training_rows
    rows = [dict(name=f'{i}.mat', role='train', size=1) for i in range(19)]
    rows[0]['name'] = name
    with pytest.raises(ValueError, match='filename'):
        training_rows(rows)
```

</details>
