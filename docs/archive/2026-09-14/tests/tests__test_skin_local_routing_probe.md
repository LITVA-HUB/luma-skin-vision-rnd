# `tests/test_skin_local_routing_probe.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_local_routing_probe.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `e84598c59ed15c8c9c37d5db5eb4917708a1fa319a8b8b9f4ac2a71c6a44bbd2`. Строк: **17**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
from skin_local_routing_probe import route
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_local_and_global_routing_agree_for_constant_patch_gate` | FunctionDef | См. реализацию | [L8](../../../../tests/test_skin_local_routing_probe.py#L8) |
| `test_varying_gate_need_not_commute_with_pooling` | FunctionDef | См. реализацию | [L13](../../../../tests/test_skin_local_routing_probe.py#L13) |

## Все тестовые определения (2)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_local_and_global_routing_agree_for_constant_patch_gate` · L8

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_local_and_global_routing_agree_for_constant_patch_gate():
    r=np.random.default_rng(31);h=r.normal(size=(3,5,4,3));w=r.dirichlet(np.ones(5),size=3);g=r.dirichlet(np.ones(4),size=3)
    np.testing.assert_allclose(route(h,w,g),route(h,w,np.repeat(g[:,None],5,axis=1)),atol=1e-14)
```

</details>

### `test_varying_gate_need_not_commute_with_pooling` · L13

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_varying_gate_need_not_commute_with_pooling():
    h=np.array([[[[50.,10.,20.],[70.,10.,20.]],[[70.,10.,20.],[50.,10.,20.]]]])
    w=np.array([[.5,.5]]);local=np.array([[[1.,0.],[0.,1.]]])
    np.testing.assert_allclose(route(h,w,local),[[50,10,20]])
    np.testing.assert_allclose(route(h,w,local.mean(1)),[[60,10,20]])
```

</details>
