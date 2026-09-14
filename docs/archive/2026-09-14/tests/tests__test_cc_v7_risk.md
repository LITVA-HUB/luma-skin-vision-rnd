# `tests/test_cc_v7_risk.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v7_risk.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `20e7b7c8989f8d06e77e44e4504599d12ea602a575e9a8f73677fc6ef2abff34`. Строк: **28**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import cc_v7_risk
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_streaming_reader_preserves_risk_then_calibration_alignment` | FunctionDef | См. реализацию | [L11](../../../../tests/test_cc_v7_risk.py#L11) |
| `test_risk_fitting_refuses_test_roles_and_overlapping_calibration` | FunctionDef | См. реализацию | [L20](../../../../tests/test_cc_v7_risk.py#L20) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_streaming_reader_preserves_risk_then_calibration_alignment` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_streaming_reader_preserves_risk_then_calibration_alignment(monkeypatch):
    def fake_read(path,key,indices,total):
        assert indices.tolist()==[1,3,4,7]
        return indices[:,None]*10
    monkeypatch.setattr(cc_v7_risk,"read_npz_rows",fake_read)
    actual=cc_v7_risk.read_in_role_order(Path("unused"),"gt",np.array([1,7,3,4]))
    np.testing.assert_array_equal(actual[:,0],[10,70,30,40])
```

</details>

### `test_risk_fitting_refuses_test_roles_and_overlapping_calibration` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_risk_fitting_refuses_test_roles_and_overlapping_calibration(tmp_path):
    values={"valid":np.ones(2,dtype=bool)}
    rows=[{"subset":"risk","group":"a"},{"subset":"test","group":"b"}]
    with pytest.raises(ValueError,match="wrong role"):
        cc_v7_risk.fit_heads(values,None,rows,1,tmp_path,"unused")
    rows[1]={"subset":"cal","group":"a"}
    with pytest.raises(ValueError,match="overlap"):
        cc_v7_risk.fit_heads(values,None,rows,1,tmp_path,"unused")
    assert not list(tmp_path.iterdir())
```

</details>
