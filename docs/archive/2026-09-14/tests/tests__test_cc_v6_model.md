# `tests/test_cc_v6_model.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v6_model.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `cc7ebdc1d4a4e9d6a818124a7a3d951dadeab731d45b90fc56bb75b425d9bf05`. Строк: **93**.

## Зависимости

```python
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from cc_v5_model import CorrectionEvidenceNet
from cc_v6_model import CanonicalEvidenceNet
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_none_frame_exactly_replays_old_numerics` | FunctionDef | См. реализацию | [L12](../../../../tests/test_cc_v6_model.py#L12) |
| `test_gain_equivariance_including_physical_risk` | FunctionDef | См. реализацию | [L25](../../../../tests/test_cc_v6_model.py#L25) |
| `test_invalid_rows_remain_invalid_and_finite` | FunctionDef | См. реализацию | [L37](../../../../tests/test_cc_v6_model.py#L37) |
| `test_training_risk_has_finite_gradients_in_residual_frame` | FunctionDef | См. реализацию | [L50](../../../../tests/test_cc_v6_model.py#L50) |
| `test_evaluator_returns_original_camera_rgb` | FunctionDef | См. реализацию | [L59](../../../../tests/test_cc_v6_model.py#L59) |
| `test_none_frame_training_matches_v5_update` | FunctionDef | См. реализацию | [L76](../../../../tests/test_cc_v6_model.py#L76) |

## Все тестовые определения (6)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_none_frame_exactly_replays_old_numerics` · L12

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_none_frame_exactly_replays_old_numerics():
    torch.set_num_threads(2)
    torch.manual_seed(6)
    old = CorrectionEvidenceNet(mode="transport").eval()
    new = CanonicalEvidenceNet(mode="transport", frame="none").eval()
    new.load_state_dict(old.state_dict())
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        a, b = old(x), new(x)
    for key in ("pred", "base_pred", "risk", "trajectory_actions"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
```

</details>

### `test_gain_equivariance_including_physical_risk` · L25

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_gain_equivariance_including_physical_risk():
    torch.manual_seed(7)
    model = CanonicalEvidenceNet(mode="transport", frame="sog").double().eval()
    x = torch.rand(2, 3, 64, 64, dtype=torch.float64) + .02
    gains = torch.tensor([.2, 1.3, 3.4], dtype=torch.float64)
    with torch.no_grad():
        a, b = model(x), model(x * gains[None, :, None, None])
    torch.testing.assert_close(b["pred"], F.normalize(a["pred"] * gains, dim=-1), atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(b["risk"], a["risk"], atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(b["trajectory_actions"], a["trajectory_actions"], atol=1e-9, rtol=1e-8)
```

</details>

### `test_invalid_rows_remain_invalid_and_finite` · L37

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_rows_remain_invalid_and_finite():
    model = CanonicalEvidenceNet(frame="sog").eval()
    x = torch.ones(3, 3, 32, 32)
    x[0] = 0
    x[1, 0] = float("nan")
    x[2, 1] = 0
    with torch.no_grad():
        output = model(x)
    assert not output["valid"].any()
    assert torch.isfinite(output["pred"]).all()
    assert torch.isfinite(output["risk"]).all()
```

</details>

### `test_training_risk_has_finite_gradients_in_residual_frame` · L50

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_training_risk_has_finite_gradients_in_residual_frame():
    model = CanonicalEvidenceNet(frame="sog").train()
    cache = model.encode(torch.rand(2, 3, 64, 64) + .02)
    loss = model.query(cache, cache["point_action"][:, None])["angular_risk"].mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
```

</details>

### `test_evaluator_returns_original_camera_rgb` · L59

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_evaluator_returns_original_camera_rgb():
    import numpy as np
    from cc_v6_evaluate import evaluate

    from luma_skin_vision.cc.core import reproduction
    model = CanonicalEvidenceNet(frame="sog").eval()
    x = torch.rand(2, 3, 64, 64) + .02
    x[:, 0] *= .2
    gt = F.normalize(torch.tensor([[.2, .6, .8], [.4, .9, .3]]), dim=-1)
    metrics, arrays = evaluate(model, x, gt, 2)
    with torch.no_grad():
        output = model(x)
    np.testing.assert_allclose(arrays["pred"], output["pred"].numpy(), atol=1e-6)
    np.testing.assert_allclose(arrays["reproduction"], reproduction(arrays["pred"], gt.numpy().astype(np.float64)), atol=1e-8)
    assert metrics["valid_fraction"] == 1
```

</details>

### `test_none_frame_training_matches_v5_update` · L76

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_none_frame_training_matches_v5_update():
    import copy

    from cc_v5_experiment import train_epoch as old_epoch
    from cc_v6_experiment import train_epoch as new_epoch
    torch.manual_seed(17)
    old = CorrectionEvidenceNet(mode="transport")
    new = CanonicalEvidenceNet(mode="transport", frame="none")
    new.load_state_dict(copy.deepcopy(old.state_dict()))
    x = torch.rand(2, 3, 64, 64) + .02
    gt = F.normalize(torch.rand(2, 3) + .2, dim=-1)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in (old, new)]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, 120) for o in optimizers]
    losses = [fn(m,o,s,x,gt,torch.arange(2),40,17,2,"transport_random",20)
              for fn,m,o,s in zip((old_epoch,new_epoch),(old,new),optimizers,schedulers)]
    assert losses[0] == losses[1]
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
```

</details>
