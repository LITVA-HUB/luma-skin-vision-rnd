# `tests/test_chromaseed_crossfit.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_crossfit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Whole-person teacher exclusion, matched routing, native scaling and fixed H schema.

SHA-256 исходника: `af5cadfd6a70750a977ed0718e9b5b560aebf67f76c2dc39a57f5c05af527a10`. Строк: **112**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_crossfit import fit_bank, fit_single, route, routing, teacher_fit, teacher_pool
from chromaseed_hybrid import fit_single as h_single
from chromaseed_hybrid import predict
from chromaseed_hybrid_numpy import Predictor
from test_chromaseed_projection import toy
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 16](../../../../tests/test_chromaseed_crossfit.py#L16)

```python
SETTINGS = {
    loss: {
        kind: dict(
            kind=kind,
            alpha=0.1 if loss == "norm" else 1.0,
            rho=0.5,
            power=1 if kind == "support" else 0,
        )
        for kind in ("uniform", "support")
    }
    for loss in ("norm", "perceptual")
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_routing_excludes_whole_person_and_matches_camera_counts` | FunctionDef | См. реализацию | [L30](../../../../tests/test_chromaseed_crossfit.py#L30) |
| `test_ambiguous_camera_or_singleton_group_is_rejected` | FunctionDef | См. реализацию | [L44](../../../../tests/test_chromaseed_crossfit.py#L44) |
| `test_excluded_person_cannot_change_teacher_parameters` | FunctionDef | См. реализацию | [L51](../../../../tests/test_chromaseed_crossfit.py#L51) |
| `bank` | FunctionDef | См. реализацию | [L65](../../../../tests/test_chromaseed_crossfit.py#L65) |
| `test_raw_backbone_is_exact_h_and_schema_is_unchanged` | FunctionDef | См. реализацию | [L69](../../../../tests/test_chromaseed_crossfit.py#L69) |
| `test_complete_single_fit_matches_full_bank` | FunctionDef | См. реализацию | [L94](../../../../tests/test_chromaseed_crossfit.py#L94) |
| `test_teacher_tables_are_native_lab_and_whole_person_oof` | FunctionDef | См. реализацию | [L102](../../../../tests/test_chromaseed_crossfit.py#L102) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_routing_excludes_whole_person_and_matches_camera_counts` · L30

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_routing_excludes_whole_person_and_matches_camera_counts():
    x, y, p, s, c = toy()
    people, own, matched = routing(p, c)
    for i, v in enumerate(p):
        assert people[own[i]] == v and people[matched[i]] != v
        assert c[p == people[matched[i]]][0] == c[i]
    table = np.broadcast_to(np.arange(len(people))[:, None, None], (len(people), len(p), 3)).astype(
        float
    )
    a, b = route(table, own, matched)
    np.testing.assert_array_equal(a[:, 0], own)
    np.testing.assert_array_equal(b[:, 0], matched)
```

</details>

### `test_ambiguous_camera_or_singleton_group_is_rejected` · L44

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_ambiguous_camera_or_singleton_group_is_rejected():
    with pytest.raises(ValueError):
        routing(np.array([0, 0, 1, 2]), np.array(["SLR", "ipod", "SLR", "ipod"]))
    with pytest.raises(ValueError):
        routing(np.array([0, 1, 2]), np.array(["SLR", "ipod", "ipod"]))
```

</details>

### `test_excluded_person_cannot_change_teacher_parameters` · L51

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_excluded_person_cannot_change_teacher_parameters():
    data = list(toy())
    before, _ = teacher_fit(*data, excluded=0, seeds=(17,), losses=("norm",), rank=4)
    data[0] = data[0].copy()
    data[1] = data[1].copy()
    data[0][data[2] == 0] = 0.4
    data[1][data[2] == 0] += 70
    after, _ = teacher_fit(*data, excluded=0, seeds=(17,), losses=("norm",), rank=4)
    for name in before:
        for k in before[name]:
            np.testing.assert_array_equal(before[name][k], after[name][k])
```

</details>

### `test_raw_backbone_is_exact_h_and_schema_is_unchanged` · L69

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_raw_backbone_is_exact_h_and_schema_is_unchanged(bank):
    models, teachers, tables, rec = bank
    assert len(models) == 30 and len(teachers) == 84
    assert rec["teacher_subsets"] == 14 and rec["new_correction_solutions"] == 24
    for loss in ("norm", "perceptual"):
        m, _ = h_single(*toy(), loss, 17, "raw", rank=4)
        for k in m:
            np.testing.assert_array_equal(models[f"{loss}_raw_s17"][k], m[k])
    for m in models.values():
        call = Predictor(m)
        np.testing.assert_allclose(
            [call(v) for v in toy()[0][:3]], predict(m, toy()[0][:3]), atol=2e-8, rtol=0
        )
    assert not np.array_equal(tables["out_person__norm_s17"], tables["in_matched__norm_s17"])
```

</details>

### `test_complete_single_fit_matches_full_bank` · L94

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('arm,loss,kind', [('out_person', 'norm', 'uniform'), ('in_matched', 'perceptual', 'uniform'), ('out_person', 'perceptual', 'support'), ('in_matched', 'norm', 'support')])
def test_complete_single_fit_matches_full_bank(bank, arm, loss, kind):
    m, _ = fit_single(*toy(), loss, 29, arm, SETTINGS[loss][kind], rank=4)
    expected = bank[0][f"{arm}_{loss}_{kind}_s29"]
    assert set(m) == set(expected)
    for k in m:
        np.testing.assert_array_equal(m[k], expected[k])
```

</details>

### `test_teacher_tables_are_native_lab_and_whole_person_oof` · L102

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_teacher_tables_are_native_lab_and_whole_person_oof():
    data = toy(False)
    models, tables, rec = teacher_pool(*data, seeds=(17,), losses=("norm",), rank=4)
    p = data[2]
    for j, person in enumerate(rec["people"]):
        excluded = p == person
        actual = predict(models[f"exclude{j}_norm_s17"], data[0][excluded])
        np.testing.assert_allclose(
            tables["out_person__norm_s17"][excluded], actual, atol=2e-8, rtol=0
        )
        assert not set(rec["subsets"][j]["fit_person_values"]) & {int(person)}
```

</details>
