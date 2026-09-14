# `tests/test_cc_v4_experiment.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v4_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Protocol tests; not experimental accuracy evidence.

SHA-256 исходника: `ee4c21256ce0dc5f59390d3273bdf8f1024c69aa1c53f5829d287cd822cbc1d8`. Строк: **62**.

## Зависимости

```python
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `runner` | FunctionDef | См. реализацию | [L14](../../../../tests/test_cc_v4_experiment.py#L14) |
| `test_action_design_uses_only_detached_prediction_and_independent_rng` | FunctionDef | См. реализацию | [L23](../../../../tests/test_cc_v4_experiment.py#L23) |
| `test_repeated_refinement_diagnostics_count_true_regressions` | FunctionDef | См. реализацию | [L35](../../../../tests/test_cc_v4_experiment.py#L35) |
| `test_fixed_coverage_is_actual_integer_population_and_finite` | FunctionDef | См. реализацию | [L46](../../../../tests/test_cc_v4_experiment.py#L46) |
| `test_zero_field_warmup_and_later_fixed_schedule` | FunctionDef | См. реализацию | [L58](../../../../tests/test_cc_v4_experiment.py#L58) |

## Все тестовые определения (4)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_action_design_uses_only_detached_prediction_and_independent_rng` · L23

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_action_design_uses_only_detached_prediction_and_independent_rng():
    r = runner()
    point = torch.tensor([[.2, -.1], [-.3, .4]], requires_grad=True)
    a = r.sample_actions(point, torch.Generator().manual_seed(19))
    b = r.sample_actions(point, torch.Generator().manual_seed(19))
    assert torch.equal(a, b) and not a.requires_grad
    assert a.shape == (2, 33, 2)
    assert torch.equal(a[:, 0], point.detach())
    assert (a.abs() <= 2).all()
    assert ((a[:, 1:17] - point.detach()[:, None]).abs() <= .400001).all()
```

</details>

### `test_repeated_refinement_diagnostics_count_true_regressions` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_repeated_refinement_diagnostics_count_true_regressions():
    r = runner()
    errors = np.array([[2., 1., 3., 4.], [3., 2., 2., 1.]])
    risks = np.array([[3., 2., 1., .5], [4., 3., 2., 1.]])
    result = r.refinement_summary(errors, risks)
    assert result["1_to_2"]["true_error_worsened"] == 0
    assert result["2_to_4"]["true_error_worsened"] == 1
    assert result["2_to_4"]["predicted_risk_increased"] == 0
    assert result["2_to_4"]["mean_true_delta"] == pytest.approx(1.)
```

</details>

### `test_fixed_coverage_is_actual_integer_population_and_finite` · L46

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fixed_coverage_is_actual_integer_population_and_finite():
    r = runner()
    errors = np.arange(1., 120.)
    result = r.risk_summary(errors, -errors)
    row = result["fixed"]["80"]
    assert row["accepted"] == 95 and row["coverage"] == pytest.approx(95 / 119)
    assert row["mean"] > errors.mean()
    assert len(result["curve"]) == 119
    with pytest.raises(ValueError):
        r.risk_summary(errors, np.full(119, np.nan))
```

</details>

### `test_zero_field_warmup_and_later_fixed_schedule` · L58

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_zero_field_warmup_and_later_fixed_schedule():
    r = runner()
    assert r.field_weight(0) == 0 and r.field_weight(20) == 0
    assert r.field_weight(30) == .5 and r.field_weight(40) == 1
    assert r.field_weight(120) == 1
```

</details>
