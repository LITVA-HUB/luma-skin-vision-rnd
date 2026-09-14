# `tests/test_chromaseed_compute_structure.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_compute_structure.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `5b657d96202d1107e11f4433493818d7077d4777f7dfbddf9028572317d1943a`. Строк: **48**.

## Зависимости

```python
import sys
from pathlib import Path
import pytest
from chromaseed_compute_structure import audit_variant, summarize_trace
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_real_meta_trace_matches_independent_operator_count_and_shared_weights` | FunctionDef | См. реализацию | [L11](../../../../tests/test_chromaseed_compute_structure.py#L11) |
| `test_bank_and_batch_scale_work_but_not_per_model_parameters` | FunctionDef | См. реализацию | [L24](../../../../tests/test_chromaseed_compute_structure.py#L24) |
| `test_soft_and_dynamic_masks_do_not_skip_dense_linear_layers` | FunctionDef | См. реализацию | [L32](../../../../tests/test_chromaseed_compute_structure.py#L32) |
| `test_operator_disagreement_is_an_error_instead_of_silent_undercounting` | FunctionDef | См. реализацию | [L39](../../../../tests/test_chromaseed_compute_structure.py#L39) |
| `test_invalid_workload_is_rejected` | FunctionDef | См. реализацию | [L46](../../../../tests/test_chromaseed_compute_structure.py#L46) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_real_meta_trace_matches_independent_operator_count_and_shared_weights` · L11

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_real_meta_trace_matches_independent_operator_count_and_shared_weights():
    result = audit_variant('dynamic5m', 'wide')
    assert result['parameters_per_model_including_anchor'] == 4_846_822
    assert result['linear_macs'] == 28_631_680
    assert result['operator_flops'] == {'aten.bmm': 57_263_360}
    assert result['passes'] == 4
    assert result['fixed_linear_macs'] == 11_255_168
    assert result['per_pass_linear_macs'] == [4_344_128]*4
    assert result['prefix_counterfactuals'][1]['linear_macs'] == 19_943_424
    assert result['latency_seconds'] is None
    assert result['cuda_initialized'] is False
```

</details>

### `test_bank_and_batch_scale_work_but_not_per_model_parameters` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bank_and_batch_scale_work_but_not_per_model_parameters():
    one = audit_variant('patch_small', 'linear')
    six = audit_variant('patch_small', 'linear', slots=2, batch=3)
    assert one['linear_macs'] == six['linear_macs']
    assert six['bank_linear_macs'] == 6*one['bank_linear_macs']
    assert one['parameters_per_model_including_anchor'] == six['parameters_per_model_including_anchor']
```

</details>

### `test_soft_and_dynamic_masks_do_not_skip_dense_linear_layers` · L32

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_soft_and_dynamic_masks_do_not_skip_dense_linear_layers():
    soft = audit_variant('soft_small', 'unit')
    dynamic = audit_variant('dynamic_small', 'unit')
    assert soft['trace'] == dynamic['trace']
    assert soft['linear_macs'] == dynamic['linear_macs']
```

</details>

### `test_operator_disagreement_is_an_error_instead_of_silent_undercounting` · L39

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_operator_disagreement_is_an_error_instead_of_silent_undercounting():
    with pytest.raises(ValueError, match='operator'):
        summarize_trace([dict(layer='head', macs=6, input_shape=[1, 1, 2])],
                        {'aten.bmm': 14}, recurrent=False, instances=1)
```

</details>

### `test_invalid_workload_is_rejected` · L46

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('slots,batch', [(0, 1), (1, 0), (True, 1)])
def test_invalid_workload_is_rejected(slots, batch):
    with pytest.raises(ValueError):
        audit_variant('soft_small', 'unit', slots=slots, batch=batch)
```

</details>
