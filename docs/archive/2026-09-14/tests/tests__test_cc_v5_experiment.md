# `tests/test_cc_v5_experiment.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v5_experiment.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Behavior checks for paired training, not accuracy evidence.

SHA-256 исходника: `ee36c1f1e2949fce7a766a9e68fe1bccccf1cc916b47955b7cc6ef4de72f7493`. Строк: **116**.

## Зависимости

```python
import importlib.util
import sys
from pathlib import Path
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `preserve_determinism_setting` | FunctionDef | См. реализацию | [L13](../../../../tests/test_cc_v5_experiment.py#L13) |
| `runner` | FunctionDef | См. реализацию | [L20](../../../../tests/test_cc_v5_experiment.py#L20) |
| `test_policy_sampling_detaches_center_and_preserves_paired_random_noise` | FunctionDef | См. реализацию | [L29](../../../../tests/test_cc_v5_experiment.py#L29) |
| `test_restore_copies_adam_moments_and_bn_without_mutating_common_start` | FunctionDef | См. реализацию | [L42](../../../../tests/test_cc_v5_experiment.py#L42) |
| `test_physical_gradient_matches_independent_finite_difference` | FunctionDef | См. реализацию | [L66](../../../../tests/test_cc_v5_experiment.py#L66) |
| `test_complete_field_backward_is_strictly_deterministic_on_cuda` | FunctionDef | См. реализацию | [L86](../../../../tests/test_cc_v5_experiment.py#L86) |
| `test_identity_pool_removal_preserves_v4_outputs_and_gradients` | FunctionDef | См. реализацию | [L99](../../../../tests/test_cc_v5_experiment.py#L99) |

## Все тестовые определения (5)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_policy_sampling_detaches_center_and_preserves_paired_random_noise` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_policy_sampling_detaches_center_and_preserves_paired_random_noise():
    r = runner()
    point = torch.tensor([[.1, -.2]], requires_grad=True)
    selected = torch.tensor([[.3, .2]], requires_grad=True)
    a = r.draw_actions(point, point, torch.Generator().manual_seed(7))
    b = r.draw_actions(point, selected, torch.Generator().manual_seed(7))
    assert not a.requires_grad and not b.requires_grad
    assert a.shape == b.shape == (1, 33, 2)
    assert torch.equal(a[:, 0], point.detach())
    assert torch.equal(a[:, 17:], b[:, 17:])
    assert torch.allclose(b[:, 1:17] - a[:, 1:17], torch.tensor([.2, .4]))
```

</details>

### `test_restore_copies_adam_moments_and_bn_without_mutating_common_start` · L42

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_restore_copies_adam_moments_and_bn_without_mutating_common_start():
    r = runner()
    torch.manual_seed(4)
    m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    opt = torch.optim.AdamW(m.parameters(), lr=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    x = torch.tensor([[1., 2., 4.], [2., -1., 0.]])
    m(x).square().sum().backward()
    opt.step()
    sch.step()
    initial = r.capture_state(m, opt, sch)
    digest = r.state_digest(initial)
    outcomes = []
    for _ in range(2):
        r.restore_state(initial, m, opt, sch)
        opt.zero_grad(set_to_none=True)
        m(x).square().sum().backward()
        opt.step()
        sch.step()
        outcomes.append(r.state_digest(r.capture_state(m, opt, sch)))
        assert r.state_digest(initial) == digest
    assert outcomes[0] == outcomes[1] != digest
```

</details>

### `test_physical_gradient_matches_independent_finite_difference` · L66

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_physical_gradient_matches_independent_finite_difference():
    r = runner()
    a = torch.tensor([[[.4, -.3], [-.2, .15]]], dtype=torch.double, requires_grad=True)
    gt = torch.tensor([[.12, -.08]], dtype=torch.double)
    target, derivative = r.physical_targets(gt, a, with_gradient=True)
    # Independent ratio formula, with no use of the model's analytic-cost helper.
    def cost(z):
        residual = gt[:, None] - z
        rgb = torch.stack([residual[..., 0].exp(), torch.ones_like(residual[..., 0]), residual[..., 1].exp()], -1)
        return 1 - rgb.sum(-1).square() / (3 * rgb.square().sum(-1))
    h = 1e-5
    for k in range(2):
        delta = torch.zeros_like(a)
        delta[..., k] = h
        finite = (cost(a.detach() + delta) - cost(a.detach() - delta)) / (2 * h)
        assert torch.allclose(derivative[..., k], finite, atol=1e-9, rtol=1e-7)
    assert not target['sin2'].requires_grad and not derivative.requires_grad
```

</details>

### `test_complete_field_backward_is_strictly_deterministic_on_cuda` · L86

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA determinism regression')
def test_complete_field_backward_is_strictly_deterministic_on_cuda():
    r = runner()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    model = r.CorrectionEvidenceNet('transport').cuda()
    cache = model.encode(torch.rand(2, 3, 128, 128, device='cuda'))
    actions = torch.zeros(2, 3, 2, device='cuda', requires_grad=True)
    field = model.query(cache, actions)
    derivative = torch.autograd.grad(field['sin2_risk'].sum(), actions, create_graph=True)[0]
    (field['angular_risk'].sum() + derivative.square().sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
```

</details>

### `test_identity_pool_removal_preserves_v4_outputs_and_gradients` · L99

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_identity_pool_removal_preserves_v4_outputs_and_gradients():
    from cc_v4_model import CorrectionEvidenceNet as Original
    r = runner()
    torch.set_num_threads(2)
    torch.manual_seed(8)
    original, revised = Original('transport').double(), r.CorrectionEvidenceNet('transport').double()
    revised.load_state_dict(original.state_dict())
    original.eval()
    revised.eval()
    x = torch.rand(2, 3, 128, 128, dtype=torch.double)
    a = torch.rand(2, 3, 2, dtype=torch.double) - .5
    first, second = original.query(original.encode(x), a), revised.query(revised.encode(x), a)
    assert torch.equal(first['angular_risk'], second['angular_risk'])
    first['angular_risk'].sum().backward()
    second['angular_risk'].sum().backward()
    for left, right in zip(original.parameters(), revised.parameters()):
        if left.grad is not None:
            assert torch.allclose(left.grad, right.grad, atol=1e-12, rtol=1e-10)
```

</details>
