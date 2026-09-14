# `tests/integration/test_export.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/integration/test_export.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `d7bcb1c9ca4408ffcf7ea0258c65bd5e540c9aed04963ad485af9d3e96a1a954`. Строк: **28**.

## Зависимости

```python
import numpy as np
import pytest
import torch
from luma_skin_vision.export import export_model
from luma_skin_vision.models import ColorRegressor
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_onnx_equivalence` | FunctionDef | См. реализацию | [L10](../../../../tests/integration/test_export.py#L10) |
| `test_error_model_requires_heldout` | FunctionDef | См. реализацию | [L24](../../../../tests/integration/test_export.py#L24) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_onnx_equivalence` · L10

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('method', ['baseline_c', 'baseline_c_plus', 'proposed_v1'])
def test_onnx_equivalence(tmp_path, method):
    pytest.importorskip("onnxruntime")
    model = ColorRegressor(method).eval()
    torch.manual_seed(1)
    # Avoid zero-output final layer hiding broken computation branches.
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    x = torch.rand(2, 3, 64, 64)
    aux = torch.rand(2, 12)
    result = export_model(model, tmp_path / "model.onnx", x, aux)
    assert result["max_abs_lab_difference"] < 1e-3
    assert result["max_delta_e00_difference"] < 1e-3
    assert result["onnx_bytes"] > 0
```

</details>

### `test_error_model_requires_heldout` · L24

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_error_model_requires_heldout():
    from luma_skin_vision.uncertainty import fit_error

    with pytest.raises(ValueError, match="out-of-fold"):
        fit_error(np.ones((10, 2)), np.ones(10), provenance="fitted_training")
```

</details>
