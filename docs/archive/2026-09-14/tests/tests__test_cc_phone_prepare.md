# `tests/test_cc_phone_prepare.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_phone_prepare.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `c36c3c0e8d377076a410ea75ecaf36bb8034a4176be8270629be81150633195b`. Строк: **42**.

## Зависимости

```python
import importlib.util
import sys
from pathlib import Path
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../tests/test_cc_phone_prepare.py#L8)

```python
SPEC = importlib.util.spec_from_file_location(
    'phone_prepare', Path(__file__).parents[1]/'scripts/cc_phone_prepare.py')
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `patch` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc_phone_prepare.py#L14) |
| `test_neutral_fallback_is_reference_only_and_preserves_chromaticity` | FunctionDef | См. реализацию | [L20](../../../../tests/test_cc_phone_prepare.py#L20) |
| `test_dark_or_disagreeing_reference_is_not_scored` | FunctionDef | См. реализацию | [L28](../../../../tests/test_cc_phone_prepare.py#L28) |
| `test_brightest_allowed_gray_selected_consistently` | FunctionDef | См. реализацию | [L38](../../../../tests/test_cc_phone_prepare.py#L38) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_neutral_fallback_is_reference_only_and_preserves_chromaticity` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_neutral_fallback_is_reference_only_and_preserves_chromaticity():
    patches = {'20': patch([.4, 1, .6], 1), '21': patch([.2, .4, .3]),
               '22': patch([.1, .2, .15])}
    result = prepare.reference_from_patches(patches)
    assert result['valid'] and result['selected_patch'] == 21
    np.testing.assert_allclose(result['gt'], np.array([2, 4, 3])/np.sqrt(29))
```

</details>

### `test_dark_or_disagreeing_reference_is_not_scored` · L28

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_dark_or_disagreeing_reference_is_not_scored():
    p = {'20': patch([.2, .4, .3]), '21': patch([.01, .02, .015]),
         '22': patch([.005, .01, .0075])}
    result = prepare.reference_from_patches(p)
    assert not result['valid'] and result['gt'] is None
    p['21'] = patch([.4, .2, .3])
    result = prepare.reference_from_patches(p)
    assert not result['valid'] and 'disagreement' in result['reason']
```

</details>

### `test_brightest_allowed_gray_selected_consistently` · L38

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_brightest_allowed_gray_selected_consistently():
    p = {str(i): patch(np.array([.2, .4, .3])/(i-19)) for i in (20, 21, 22)}
    result = prepare.reference_from_patches(p)
    assert result['valid'] and result['selected_patch'] == 20
    assert result['max_pairwise_degrees'] < 1e-6
```

</details>
