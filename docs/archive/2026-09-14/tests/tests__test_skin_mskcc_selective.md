# `tests/test_skin_mskcc_selective.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_mskcc_selective.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `c5a7199d5560a8ee7b740aecdcd7723b6633ab2f206bac52425c213a6c6f2b34`. Строк: **58**.

## Зависимости

```python
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pytest
import skin_mskcc_selective_core as core
import skin_mskcc_selective_data as reserved
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_grouped_oof_assignment_covers_each_person_once` | FunctionDef | См. реализацию | [L10](../../../../tests/test_skin_mskcc_selective.py#L10) |
| `test_sealed_data_checks_lock_before_any_endpoint_io` | FunctionDef | См. реализацию | [L17](../../../../tests/test_skin_mskcc_selective.py#L17) |
| `test_lock_checks_stage_and_bound_bytes` | FunctionDef | См. реализацию | [L24](../../../../tests/test_skin_mskcc_selective.py#L24) |
| `test_risk_feature_contract_keeps_common_inputs_identical` | FunctionDef | См. реализацию | [L34](../../../../tests/test_skin_mskcc_selective.py#L34) |
| `test_instrument_reference_retains_actual_single_measurement_without_imputation` | FunctionDef | См. реализацию | [L43](../../../../tests/test_skin_mskcc_selective.py#L43) |
| `test_instrument_reference_rejects_missing_and_inconsistent_ground_truth` | FunctionDef | См. реализацию | [L53](../../../../tests/test_skin_mskcc_selective.py#L53) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_grouped_oof_assignment_covers_each_person_once` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_grouped_oof_assignment_covers_each_person_once():
    people=[f'p{i}' for i in range(24)]
    folds=core.patient_folds(people)
    assert folds==core.patient_folds(people[::-1])
    assert sorted(list(folds.values()).count(i) for i in range(6))==[4]*6
```

</details>

### `test_sealed_data_checks_lock_before_any_endpoint_io` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_sealed_data_checks_lock_before_any_endpoint_io(monkeypatch):
    def fail(*args,**kwargs):raise ValueError('wrong lock')
    def io():pytest.fail('Endpoint I/O before lock verification')
    monkeypatch.setattr(reserved,'verify_lock',fail);monkeypatch.setattr(reserved,'manifest',io)
    with pytest.raises(ValueError,match='wrong lock'):reserved.sealed('test','missing','bad')
```

</details>

### `test_lock_checks_stage_and_bound_bytes` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_lock_checks_stage_and_bound_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'ROOT',tmp_path)
    value=tmp_path/'value';value.write_bytes(b'original')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'stage':'final','bindings':{'value':core.sha(value)}}))
    digest=core.sha(lock);core.verify_lock(lock,digest,'final')
    with pytest.raises(ValueError,match='stage'):core.verify_lock(lock,digest,'precalibration')
    value.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):core.verify_lock(lock,digest,'final')
```

</details>

### `test_risk_feature_contract_keeps_common_inputs_identical` · L34

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_risk_feature_contract_keeps_common_inputs_identical():
    data={'color':np.ones((5,36))};p=np.arange(45).reshape(3,5,3);w=np.ones((3,5,8))
    d=core.designs(data,p,w,np.arange(5))
    for r in d.values():
        np.testing.assert_array_equal(r['C_plus'],r['Proposed'][:,:40])
    assert np.all(d['single17']['Proposed'][:,-1]==0)
    assert np.all(d['ensemble']['Proposed'][:,-1]>0)
```

</details>

### `test_instrument_reference_retains_actual_single_measurement_without_imputation` · L43

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_instrument_reference_retains_actual_single_measurement_without_imputation():
    row={f'{c}_{i}':str(v) if i==1 else '' for c,v in zip('lab',[50,12,20]) for i in [1,2,3]}
    row.update(dict(zip(['average_l','average_a','average_b'],['50','12','20'])))
    target,rep=reserved.instrument_reference(row)
    np.testing.assert_array_equal(target,[50,12,20])
    assert np.isnan(rep[1:]).all()
    row['l_2']='51'
    with pytest.raises(ValueError,match='Partial'):reserved.instrument_reference(row)
```

</details>

### `test_instrument_reference_rejects_missing_and_inconsistent_ground_truth` · L53

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_instrument_reference_rejects_missing_and_inconsistent_ground_truth():
    row={f'{c}_{i}':'' for c in 'lab' for i in [1,2,3]}
    row.update({f'average_{c}':'0' for c in 'lab'})
    with pytest.raises(ValueError,match='No complete'):reserved.instrument_reference(row)
    for c in 'lab':row[f'{c}_1']='1'
    with pytest.raises(ValueError,match='inconsistent'):reserved.instrument_reference(row)
```

</details>
