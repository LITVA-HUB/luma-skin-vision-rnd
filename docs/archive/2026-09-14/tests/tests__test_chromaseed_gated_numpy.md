# `tests/test_chromaseed_gated_numpy.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_gated_numpy.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `049576e322430d620b94f9cab14f529834e0ee2360e913fed863a12594609961`. Строк: **61**.

## Зависимости

```python
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 8](../../../../tests/test_chromaseed_gated_numpy.py#L8)

```python
ROOT = Path(__file__).resolve().parents[1]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `payload` | FunctionDef | См. реализацию | [L12](../../../../tests/test_chromaseed_gated_numpy.py#L12) |
| `test_hard_gate_boundary_and_soft_saturation_have_known_answers` | FunctionDef | См. реализацию | [L28](../../../../tests/test_chromaseed_gated_numpy.py#L28) |
| `test_invalid_payload_and_input_fail_before_arithmetic` | FunctionDef | См. реализацию | [L45](../../../../tests/test_chromaseed_gated_numpy.py#L45) |
| `test_standalone_module_does_not_import_training_frameworks` | FunctionDef | См. реализацию | [L59](../../../../tests/test_chromaseed_gated_numpy.py#L59) |

## Все тестовые определения (3)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_hard_gate_boundary_and_soft_saturation_have_known_answers` · L28

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_hard_gate_boundary_and_soft_saturation_have_known_answers():
    from chromaseed_gated_numpy import Predictor

    model = payload()
    hard = Predictor(model)
    for value in (-2.0, -1e-9, 0.0, 1e-9, 2.0):
        x = np.zeros(36, np.float32)
        x[0] = value
        expected = np.exp(-0.5 * float(x[0]) ** 2 / 36) * (1.0 + (0.5 if value >= 0 else -0.5))
        np.testing.assert_allclose(hard(x), np.full(3, expected), atol=1e-12, rtol=0)
    model["gate_mode"] = np.array(1, np.uint8)
    soft = Predictor(model)
    x = np.zeros(36, np.float32)
    x[0] = 2.0
    np.testing.assert_allclose(soft(x), hard(x), atol=1e-12, rtol=0)
```

</details>

### `test_invalid_payload_and_input_fail_before_arithmetic` · L45

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_payload_and_input_fail_before_arithmetic():
    from chromaseed_gated_numpy import Predictor

    model = payload()
    model["gate_beta"] = np.zeros(36, np.float32)
    with pytest.raises(ValueError):
        Predictor(model)
    call = Predictor(payload())
    with pytest.raises(ValueError):
        call(np.zeros(35))
    with pytest.raises(ValueError):
        call(np.full(36, np.nan))
```

</details>

### `test_standalone_module_does_not_import_training_frameworks` · L59

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_standalone_module_does_not_import_training_frameworks():
    code = "import sys; sys.path.insert(0, 'scripts'); import chromaseed_gated_numpy; assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)
```

</details>
