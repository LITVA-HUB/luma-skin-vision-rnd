# `tests/test_skin_risk_cross.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_risk_cross.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `4c0ca4d28297fc56bc255a796494221b51c8252ccbf06f9789d5e03ad9fb75dd`. Строк: **22**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_risk_cross import expected_error
from skin_mskcc_audit import scalar_de
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_zero_variance_risk_is_actual_color_distance` | FunctionDef | См. реализацию | [L9](../../../../tests/test_skin_risk_cross.py#L9) |
| `test_risk_averages_each_image_separately` | FunctionDef | См. реализацию | [L17](../../../../tests/test_skin_risk_cross.py#L17) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_zero_variance_risk_is_actual_color_distance` · L9

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_zero_variance_risk_is_actual_color_distance():
    pred=np.array([[40,14,20],[70,10,15]],float)
    center=np.array([[50,14,20],[70,10,15]],float)
    z=np.array([[-1,-1,-1],[1,1,1]],float)
    result=expected_error(pred,center,np.zeros_like(center),z)
    np.testing.assert_allclose(result,[scalar_de(a,b) for a,b in zip(pred,center)],atol=1e-12)
```

</details>

### `test_risk_averages_each_image_separately` · L17

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_risk_averages_each_image_separately():
    p=np.array([[50,10,20],[65,12,25]],float);s=np.array([[2,1,3],[4,3,2]],float)
    z=np.array([[-1,-1,-1],[1,1,1]],float)
    out=expected_error(p,p,s,z)
    expected=[np.mean([scalar_de(c,c+sigma*n) for n in z]) for c,sigma in zip(p,s)]
    np.testing.assert_allclose(out,expected,atol=1e-12)
```

</details>
