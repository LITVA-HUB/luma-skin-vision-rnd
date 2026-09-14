# `tests/test_chromaseed_crossfit_reference.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_crossfit_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent native teacher decoding and QR correction reconstruction.

SHA-256 исходника: `0b689be5b560cbd4961ccce6edc440eec2efd60cb50014ba9ca170b21bdc660b`. Строк: **47**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_crossfit import fit_bank
from chromaseed_crossfit_reference import head, teacher
from chromaseed_hybrid_reference import Basis, Geometry, predict
from test_chromaseed_crossfit import SETTINGS
from test_chromaseed_projection import toy
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_independent_teacher_and_native_residual_heads` | FunctionDef | См. реализацию | [L18](../../../../tests/test_chromaseed_crossfit_reference.py#L18) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_independent_teacher_and_native_residual_heads` · L18

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('two', [True, False])
def test_independent_teacher_and_native_residual_heads(two):
    data = toy(two)
    models, teachers, tables, rec = fit_bank(*data, SETTINGS, rank=4)
    geometry = Geometry(*data)
    basis = Basis(geometry, 17, rank=4)
    p = data[2]
    native = []
    for j, person in enumerate(rec["people"]):
        keep = p != person
        ref, _ = teacher(
            teachers[f"exclude{j}_perceptual_s17"],
            *(a[keep] for a in data),
            "perceptual",
            17,
            rank=4,
        )
        native.append(predict(ref, data[0]))
    native = np.stack(native)
    np.testing.assert_allclose(native, tables["native__perceptual_s17"], atol=0.001, rtol=0)
    for arm, indices in (
        ("out_person", tables["own_teacher"]),
        ("in_matched", tables["matched_teacher"]),
    ):
        routed = native[indices, np.arange(len(p))]
        for kind in ("uniform", "support"):
            m = head(basis, routed, "perceptual", SETTINGS["perceptual"][kind])
            expected = models[f"{arm}_perceptual_{kind}_s17"]
            np.testing.assert_allclose(
                predict(m, data[0]), predict(expected, data[0]), atol=0.001, rtol=0
            )
```

</details>
