# `tests/test_skin_mskcc.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_mskcc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `8858f03cb285ba5f455b9e1be8fb54227bd31643ae6ccaefc838c0c906a1261c`. Строк: **26**.

## Зависимости

```python
import importlib.util
from pathlib import Path
import pytest
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 5](../../../../tests/test_skin_mskcc.py#L5)

```python
SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/skin_mskcc_data.py'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_patient_partition_is_order_independent_and_exact` | FunctionDef | См. реализацию | [L11](../../../../tests/test_skin_mskcc.py#L11) |
| `test_source_loader_denies_held_out_endpoints_before_io` | FunctionDef | См. реализацию | [L21](../../../../tests/test_skin_mskcc.py#L21) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_patient_partition_is_order_independent_and_exact` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_patient_partition_is_order_independent_and_exact():
    devices = {**{f's{i}': 'SLR' for i in range(18)}, **{f'p{i}': 'ipod' for i in range(28)}}
    roles = data.patient_roles(devices)
    assert roles == data.patient_roles(dict(reversed(list(devices.items()))))
    assert {r: list(roles.values()).count(r) for r in set(roles.values())} == {
        'train': 24, 'validation': 6, 'calibration': 6, 'test': 10}
    assert sum(r == 'test' and devices[p] == 'SLR' for p, r in roles.items()) == 4
```

</details>

### `test_source_loader_denies_held_out_endpoints_before_io` · L21

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('role', ['test', 'calibration', 'all'])
def test_source_loader_denies_held_out_endpoints_before_io(role, monkeypatch):
    def forbidden():
        pytest.fail('Endpoint guard performed I/O first')
    monkeypatch.setattr(data, 'manifest', forbidden)
    with pytest.raises(ValueError, match='cannot open'):
        data.load_source(role)
```

</details>
