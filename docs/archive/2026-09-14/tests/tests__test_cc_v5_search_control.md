# `tests/test_cc_v5_search_control.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v5_search_control.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `1e036659007b13a591e0c6243c864ac4a4911c627e9d333f79106dad7601bec7`. Строк: **36**.

## Зависимости

```python
import importlib.util
from pathlib import Path
import torch
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 6](../../../../tests/test_cc_v5_search_control.py#L6)

```python
SPEC = importlib.util.spec_from_file_location('search_control', Path(__file__).parents[1]/'scripts/cc_v5_search_control.py')
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Quadratic` | — | [L11](../../../../tests/test_cc_v5_search_control.py#L11) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `Quadratic` | ClassDef | См. реализацию | [L11](../../../../tests/test_cc_v5_search_control.py#L11) |
| `test_fixed_multiscale_is_exact_budget_and_never_recenters` | FunctionDef | См. реализацию | [L22](../../../../tests/test_cc_v5_search_control.py#L22) |
| `test_refused_input_does_not_take_arbitrary_corner` | FunctionDef | См. реализацию | [L32](../../../../tests/test_cc_v5_search_control.py#L32) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_fixed_multiscale_is_exact_budget_and_never_recenters` · L22

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fixed_multiscale_is_exact_budget_and_never_recenters():
    model = Quadratic()
    cache = {'point_action': torch.zeros(1, 2), 'valid': torch.ones(1, dtype=torch.bool)}
    for steps, count in ((1, 25), (2, 51), (4, 103)):
        model.queries = 0
        result = module.fixed_select(model, cache, steps)
        assert model.queries == count == result['query_count']
        torch.testing.assert_close(result['action'], torch.tensor([[.24, .24]]))
```

</details>

### `test_refused_input_does_not_take_arbitrary_corner` · L32

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_refused_input_does_not_take_arbitrary_corner():
    cache = {'point_action': torch.ones(1, 2), 'valid': torch.zeros(1, dtype=torch.bool)}
    result = module.fixed_select(Quadratic(), cache, 2)
    assert not result['valid'].any()
    torch.testing.assert_close(result['action'], torch.zeros(1, 2))
```

</details>
