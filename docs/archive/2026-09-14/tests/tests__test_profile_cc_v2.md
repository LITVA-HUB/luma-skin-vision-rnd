# `tests/test_profile_cc_v2.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_profile_cc_v2.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Portable inference engineering checks, with synthetic pixels and fitted heads.

SHA-256 исходника: `9d586446f79580079d5c0c5802dc4ddb8397a3f4a073e6a61037f64e626dc64c`. Строк: **73**.

## Зависимости

```python
import importlib.util
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

## Классы и наследование

Классы включают сети, потребителей, датаклассы и служебные объекты. Это не счётчик независимых архитектур.

| Класс | Базовые классы | Исходник |
|---|---|---|
| `Toy` | torch.nn.Module | [L21](../../../../tests/test_profile_cc_v2.py#L21) |

## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `api` | FunctionDef | См. реализацию | [L13](../../../../tests/test_profile_cc_v2.py#L13) |
| `Toy` | ClassDef | См. реализацию | [L21](../../../../tests/test_profile_cc_v2.py#L21) |
| `fixture` | FunctionDef | См. реализацию | [L27](../../../../tests/test_profile_cc_v2.py#L27) |
| `test_embedded_ridge_matches_sklearn_and_rejects_invalid` | FunctionDef | См. реализацию | [L35](../../../../tests/test_profile_cc_v2.py#L35) |
| `test_portable_onnx_dynamic_batch_and_invalid_contract` | FunctionDef | См. реализацию | [L61](../../../../tests/test_profile_cc_v2.py#L61) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_embedded_ridge_matches_sklearn_and_rejects_invalid` · L35

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_embedded_ridge_matches_sklearn_and_rejects_invalid():
    module = api()
    torch.set_num_threads(4)
    payload = fixture()
    wrapper = module.PortableCC(Toy(), payload, threshold=100, head_dtype=torch.float64).eval()
    x = torch.rand(4, 3, 128, 128)
    with torch.inference_mode():
        pred, score, valid, accept = wrapper(x)
        p, context = wrapper.model(x)
        feature = module.risk_features_invariant(x, p, context)["combined"].numpy()
    expected = (
        module.selector.raw_predict(payload["model"], feature.astype(np.float64)) * payload["scale"]
    )
    np.testing.assert_allclose(score.numpy(), expected, rtol=1e-12, atol=1e-12)
    assert valid.all() and accept.all() and torch.isfinite(pred).all()
    invalid = x.clone()
    invalid[0] = 0
    invalid[1, 1] = 0
    invalid[2, 0, 0, 0] = float("nan")
    invalid[3, 0, 0, 0] = -1
    with torch.inference_mode():
        pred, score, valid, accept = wrapper(invalid)
    assert not valid.any() and not accept.any()
    assert torch.isfinite(pred).all() and torch.isfinite(score).all()
```

</details>

### `test_portable_onnx_dynamic_batch_and_invalid_contract` · L61

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_portable_onnx_dynamic_batch_and_invalid_contract(tmp_path):
    module = api()
    wrapper = module.PortableCC(Toy(), fixture(), threshold=100).eval()
    rows = torch.rand(4, 3, 128, 128)
    report = module.export_and_check(wrapper, rows, tmp_path / "portable.onnx")
    assert report["passed"]
    assert report["floating_dtypes"] == ["FLOAT"]
    assert set(report["cases"]) == {
        "real_batch1",
        "real_batch4",
        "invalid_batch4",
        "infinite_batch1",
    }
```

</details>
