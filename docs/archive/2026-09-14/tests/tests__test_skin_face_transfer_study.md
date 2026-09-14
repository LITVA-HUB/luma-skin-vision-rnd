# `tests/test_skin_face_transfer_study.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_face_transfer_study.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `21c4cd47101d16c2b555fccfe2e1cc0ab738df49eb3b255a59ab698baaca3716`. Строк: **91**.

## Зависимости

```python
import copy
import sys
from pathlib import Path
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `row` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_face_transfer_study.py#L10) |
| `test_selection_respects_budget_initialization_and_earliest_ties` | FunctionDef | См. реализацию | [L17](../../../../tests/test_skin_face_transfer_study.py#L17) |
| `test_invalid_selection_inputs_are_rejected` | FunctionDef | См. реализацию | [L29](../../../../tests/test_skin_face_transfer_study.py#L29) |
| `test_recipe_and_all_eighteen_choices_are_complete_and_stable` | FunctionDef | См. реализацию | [L50](../../../../tests/test_skin_face_transfer_study.py#L50) |
| `test_selection_copies_records_and_does_not_mutate_inputs` | FunctionDef | См. реализацию | [L69](../../../../tests/test_skin_face_transfer_study.py#L69) |
| `test_write_once_never_replaces_and_rehash_detects_changes` | FunctionDef | См. реализацию | [L78](../../../../tests/test_skin_face_transfer_study.py#L78) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_selection_respects_budget_initialization_and_earliest_ties` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_respects_budget_initialization_and_earliest_ties():
    from skin_face_transfer_study import select_prefix
    history = [row(0), row(498, .7, .5), row(996, .8, .6), row(1494, .9, .7),
               row(1992, .95, .85)]
    assert select_prefix(history, 996)['step'] == 0
    assert select_prefix(history, 1494)['step'] == 1494
    assert select_prefix(history, 1992)['step'] == 1992
    # Equal source weighting: many more CelebA rows must not weight it more.
    assert select_prefix([row(0, .95, .55), row(498, .6, .8)], 498)['step'] == 0
```

</details>

### `test_invalid_selection_inputs_are_rejected` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('defect', ['nan', 'missing', 'duplicate', 'unordered', 'no_zero', 'range', 'test'])
def test_invalid_selection_inputs_are_rejected(defect):
    from skin_face_transfer_study import select_prefix
    history = [row(0), row(498)]
    if defect == 'nan':
        history[1]['validation']['lapa']['mean_image_iou'] = float('nan')
    elif defect == 'missing':
        del history[0]['validation']['celeba']
    elif defect == 'duplicate':
        history.append(row(498))
    elif defect == 'unordered':
        history = history[::-1]
    elif defect == 'no_zero':
        history = history[1:]
    elif defect == 'range':
        history[0]['validation']['celeba']['mean_image_iou'] = 1.1
    else:
        history[1]['test'] = {'lapa': .99}
    with pytest.raises(ValueError):
        select_prefix(history, 498)
```

</details>

### `test_recipe_and_all_eighteen_choices_are_complete_and_stable` · L50

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_recipe_and_all_eighteen_choices_are_complete_and_stable():
    from skin_face_transfer_study import freeze_choices, recipe
    spec = recipe()
    assert spec['budgets'] == [1494, 2988, 5976]
    assert spec['batch_size'] == 32 and spec['validation_interval'] == 498
    assert spec['new_updates'] == 35856 and spec['new_image_presentations'] == 1147392
    assert [(r['arm'], r['seed']) for r in spec['trajectories']] == [
        (a, s) for a in ('lapa_only', 'lapa_celeba') for s in (17, 29, 43)]
    histories = {r['id']:[row(s) for s in range(0, 5977, 498)] for r in spec['trajectories']}
    selected = freeze_choices(histories)
    assert len(selected['choices']) == 18
    assert all(c['step'] == 0 for c in selected['choices'])
    assert selected['overall']['arm'] == 'lapa_only' and selected['overall']['seed'] == 17
    assert selected['test_accessed'] is False
    histories[spec['trajectories'][-1]['id']] = histories[spec['trajectories'][-1]['id']][:-1]
    with pytest.raises(ValueError, match='complete'):
        freeze_choices(histories)
```

</details>

### `test_selection_copies_records_and_does_not_mutate_inputs` · L69

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_copies_records_and_does_not_mutate_inputs():
    from skin_face_transfer_study import select_prefix
    history = [row(0), row(498, .9, .8)]
    before = copy.deepcopy(history)
    choice = select_prefix(history, 498)
    choice['checkpoint']['sha256'] = 'mutated'
    assert history == before
```

</details>

### `test_write_once_never_replaces_and_rehash_detects_changes` · L78

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_write_once_never_replaces_and_rehash_detects_changes(tmp_path):
    from skin_face_transfer_data import digest
    from skin_face_transfer_study import check_bindings, write_once
    path = tmp_path / 'receipt.json'
    write_once(path, {'message': 'Проверка'})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_once(path, {'message': 'Замена'})
    assert path.read_bytes() == original
    bindings = {str(path): digest(path)}
    check_bindings(bindings)
    path.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed'):
        check_bindings(bindings)
```

</details>
