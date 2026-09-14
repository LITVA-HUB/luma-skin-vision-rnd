# `tests/test_chromaseed_fast_kernel_protocol.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_fast_kernel_protocol.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `5871e374949c5aeff96c60b1c9088c25fe796479b784aed6a0f7702e6c931929`. Строк: **41**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_bank_query_matches_actual_predictors` | FunctionDef | См. реализацию | [L9](../../../../tests/test_chromaseed_fast_kernel_protocol.py#L9) |
| `test_fast_policy_requires_both_mean_and_tail_and_has_exact_fallback` | FunctionDef | См. реализацию | [L22](../../../../tests/test_chromaseed_fast_kernel_protocol.py#L22) |
| `test_size_policy_checks_tail_against_best_mean_rank` | FunctionDef | См. реализацию | [L34](../../../../tests/test_chromaseed_fast_kernel_protocol.py#L34) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_bank_query_matches_actual_predictors` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_bank_query_matches_actual_predictors():
    from chromaseed_fast_kernel import evaluate_bank, fit_bank
    from chromaseed_kernel import predict_kernel
    rng = np.random.default_rng(440)
    x, y = rng.normal(size=(49, 36)).astype(np.float32), rng.normal(size=(49, 3))
    models, receipt = fit_bank(x, y, rng.uniform(.5, 2, size=len(x)))
    assert len(models) == receipt["readout_configurations"] == 405
    q = rng.normal(size=(7, 36)).astype(np.float32)
    result = evaluate_bank(models, q, np.arange(len(q)))
    for key in list(models)[::19]:
        np.testing.assert_allclose(result[f"pred__{key}"], predict_kernel(models[key], q), rtol=0, atol=2e-9)
```

</details>

### `test_fast_policy_requires_both_mean_and_tail_and_has_exact_fallback` · L22

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fast_policy_requires_both_mean_and_tail_and_has_exact_fallback():
    from chromaseed_fast_kernel_train import choose_fast
    candidates = [{"arm": "column_exact", "person_mean": 5., "p90": 10.},
                  {"arm": "column_pairs1024", "person_mean": 5.01, "p90": 10.3},
                  {"arm": "column_pairs4096", "person_mean": 5.02, "p90": 10.04},
                  {"arm": "column_pairs16384", "person_mean": 4.99, "p90": 9.9}]
    assert choose_fast(candidates)["arm"] == "column_pairs4096"
    for candidate in candidates[1:]:
        candidate["person_mean"] = 5.2
    assert choose_fast(candidates)["arm"] == "column_exact"
```

</details>

### `test_size_policy_checks_tail_against_best_mean_rank` · L34

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_size_policy_checks_tail_against_best_mean_rank():
    from chromaseed_fast_kernel_train import choose_size
    candidates = [{"rank": 64, "person_mean": 5.03, "p90": 10.15},
                  {"rank": 128, "person_mean": 5., "p90": 10.},
                  {"rank": 256, "person_mean": 5.01, "p90": 9.8}]
    assert choose_size(candidates)["rank"] == 128
    candidates[0]["p90"] = 10.05
    assert choose_size(candidates)["rank"] == 64
```

</details>
