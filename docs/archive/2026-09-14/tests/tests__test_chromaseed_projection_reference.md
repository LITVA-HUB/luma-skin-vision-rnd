# `tests/test_chromaseed_projection_reference.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_projection_reference.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Independent refit covers identity, compact and full-rotation paths.

SHA-256 исходника: `9e9273e9cf703d05a5320db26bf9969cdbcae2f1aeaf3b4495761f251bde7369`. Строк: **27**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
from chromaseed_projection import fit_single, predict
from chromaseed_projection_reference import predict as reference_predict
from chromaseed_projection_reference import refit
from test_chromaseed_projection import toy
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_svd_projection_dense_landmarks_qr_readout_matches_primary` | FunctionDef | См. реализацию | [L20](../../../../tests/test_chromaseed_projection_reference.py#L20) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_svd_projection_dense_landmarks_qr_readout_matches_primary` · L20

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('representation,family', [('raw', 'norm_static'), ('d8_t01', 'perceptual_joint_soft'), ('d36_t1', 'norm_joint_soft')])
def test_svd_projection_dense_landmarks_qr_readout_matches_primary(representation, family):
    data = toy()
    model, _ = fit_single(*data, family, 17, representation, 0.1, rank=6)
    reference, info = refit(model, *data, family, 0.1, representation, 17, rank=6)
    assert info["actual_centers"] == 6
    np.testing.assert_allclose(
        reference_predict(reference, data[0]), predict(model, data[0]), rtol=0, atol=0.001
    )
```

</details>
