# `tests/test_skin_face_transfer_run.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_face_transfer_run.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `849bf9ca1e455c75b466001c3758b182177d2e52d93cb5ddf8dd4bed61b0b96e`. Строк: **126**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_checkpoint_round_trip_uses_exact_finite_tensor_payload` | FunctionDef | См. реализацию | [L12](../../../../tests/test_skin_face_transfer_run.py#L12) |
| `test_invalid_initialization_never_falls_back_to_random` | FunctionDef | См. реализацию | [L39](../../../../tests/test_skin_face_transfer_run.py#L39) |
| `test_gpu_gate_refuses_live_unknown_or_unsealed_worker_before_setup` | FunctionDef | См. реализацию | [L49](../../../../tests/test_skin_face_transfer_run.py#L49) |
| `test_nonfinite_gradient_fails_without_optimizer_update` | FunctionDef | См. реализацию | [L63](../../../../tests/test_skin_face_transfer_run.py#L63) |
| `test_tensor_input_preserves_rgb_channels_and_binary_targets` | FunctionDef | См. реализацию | [L78](../../../../tests/test_skin_face_transfer_run.py#L78) |
| `test_progress_is_optional_but_does_not_destroy_old_record` | FunctionDef | См. реализацию | [L89](../../../../tests/test_skin_face_transfer_run.py#L89) |
| `test_completed_stage_requires_actual_dead_worker` | FunctionDef | См. реализацию | [L102](../../../../tests/test_skin_face_transfer_run.py#L102) |
| `test_preflight_receipt_cannot_unlock_production_while_its_process_is_live` | FunctionDef | См. реализацию | [L115](../../../../tests/test_skin_face_transfer_run.py#L115) |

## Все тестовые определения (8)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_checkpoint_round_trip_uses_exact_finite_tensor_payload` · L12

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_checkpoint_round_trip_uses_exact_finite_tensor_payload(tmp_path):
    from skin_face_segment import SkinUNet
    from skin_face_transfer_data import digest
    from skin_face_transfer_run import load_checkpoint, save_checkpoint
    torch.manual_seed(7)
    model = SkinUNet().eval()
    path = tmp_path / 'own.pt'
    descriptor = save_checkpoint(path, model, {'step': 498, 'arm': 'lapa_only', 'seed': 17,
                                             'registration_sha256': 'a'*64})
    loaded = load_checkpoint(descriptor)
    for name, actual in loaded.state_dict().items():
        assert torch.equal(actual, model.state_dict()[name])
    x = torch.rand(1, 3, 16, 16)
    with torch.inference_mode():
        assert torch.equal(model(x), loaded(x))
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, {'step': 498})
    with pytest.raises(ValueError, match='Changed'):
        load_checkpoint({**descriptor, 'sha256': 'f'*64})
    payload = torch.load(path, weights_only=True)
    payload['state_dict'].pop(next(iter(payload['state_dict'])))
    broken = tmp_path / 'broken.pt'
    torch.save(payload, broken)
    with pytest.raises((RuntimeError, ValueError)):
        load_checkpoint({**descriptor, 'path': str(broken), 'sha256': digest(broken)})
```

</details>

### `test_invalid_initialization_never_falls_back_to_random` · L39

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_invalid_initialization_never_falls_back_to_random(tmp_path):
    from skin_face_transfer_run import load_initial
    with pytest.raises(FileNotFoundError):
        load_initial(tmp_path/'missing.pt', 'a'*64)
    path = tmp_path/'garbage.pt'
    path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='Changed'):
        load_initial(path, 'a'*64)
```

</details>

### `test_gpu_gate_refuses_live_unknown_or_unsealed_worker_before_setup` · L49

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_gpu_gate_refuses_live_unknown_or_unsealed_worker_before_setup(tmp_path):
    from skin_face_transfer_run import check_predecessor_records
    from skin_face_transfer_study import write_once
    job, seal = tmp_path/'job.json', tmp_path/'verification.json'
    write_once(job, {'status': 'complete', 'pid': 987654})
    for state in (True, None):
        with pytest.raises(RuntimeError, match='worker'):
            check_predecessor_records([(job, seal)], probe=lambda _: state)
    with pytest.raises(RuntimeError, match='seal'):
        check_predecessor_records([(job, seal)], probe=lambda _: False)
    assert not torch.cuda.is_initialized()
    assert not seal.exists()
```

</details>

### `test_nonfinite_gradient_fails_without_optimizer_update` · L63

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_nonfinite_gradient_fails_without_optimizer_update():
    from skin_face_segment import SkinUNet
    from skin_face_transfer_run import update
    model = SkinUNet(width=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    original = {k:v.clone() for k,v in model.state_dict().items()}
    handle = next(model.parameters()).register_hook(lambda grad: grad * float('nan'))
    with pytest.raises((RuntimeError, ValueError), match='nonfinite|finite'):
        update(model, optimizer, None, torch.rand(2,3,16,16), torch.zeros(2,1,16,16), 0)
    handle.remove()
    assert not optimizer.state
    for k,v in model.state_dict().items():
        assert torch.equal(v, original[k])
```

</details>

### `test_tensor_input_preserves_rgb_channels_and_binary_targets` · L78

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_tensor_input_preserves_rgb_channels_and_binary_targets():
    from skin_face_transfer_run import tensor_batch
    rgb = np.zeros((2,16,16,3), np.uint8)
    rgb[0,...] = [255,128,0]
    labels = np.zeros((2,16,16), np.uint8)
    labels[1,4:8,4:8] = 1
    x,y = tensor_batch(rgb, labels, 'cpu')
    torch.testing.assert_close(x[0,:,0,0], torch.tensor([1.,128/255,0.]), rtol=0, atol=0)
    assert y.shape == (2,1,16,16) and int(y.sum()) == 16
```

</details>

### `test_progress_is_optional_but_does_not_destroy_old_record` · L89

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_progress_is_optional_but_does_not_destroy_old_record(tmp_path, monkeypatch):
    import skin_face_transfer_run as run
    from skin_face_transfer_data import read
    path = tmp_path/'progress.json'
    assert run.progress(path, {'step': 0})
    def locked(*args):
        raise PermissionError('temporary Windows reader')
    monkeypatch.setattr(run.os, 'replace', locked)
    assert run.progress(path, {'step': 1}, retries=2, delay=0) is False
    assert read(path) == {'step': 0}
    assert list(tmp_path.iterdir()) == [path]
```

</details>

### `test_completed_stage_requires_actual_dead_worker` · L102

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_completed_stage_requires_actual_dead_worker(tmp_path,monkeypatch):
    import skin_face_transfer_run as run
    from skin_face_transfer_study import write_once
    path = tmp_path/'completion.json'
    write_once(path,{'status':'complete','pid':987654})
    for state in (True,None):
        monkeypatch.setattr(run,'process_alive',lambda _:state)
        with pytest.raises(RuntimeError,match='worker'):
            run.require_finished(path)
    monkeypatch.setattr(run,'process_alive',lambda _:False)
    assert run.require_finished(path)['status'] == 'complete'
```

</details>

### `test_preflight_receipt_cannot_unlock_production_while_its_process_is_live` · L115

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_preflight_receipt_cannot_unlock_production_while_its_process_is_live(tmp_path,monkeypatch):
    import skin_face_transfer_run as run
    from skin_face_transfer_study import write_once
    path = tmp_path/'preflights/cpu_fixture/result.json'
    write_once(path,{'status':'passed','device':'cpu','registration_sha256':'a'*64,'pid':987654,
                     'successful_updates':12,'cases':[{} for _ in range(6)],'anchor_pairs_exact':True,'batch_size':2})
    monkeypatch.setattr(run,'RUN',tmp_path)
    monkeypatch.setattr(run,'process_alive',lambda _:True)
    with pytest.raises(RuntimeError,match='preflight'):
        run.matching_preflight('cpu','a'*64)
    monkeypatch.setattr(run,'process_alive',lambda _:False)
    assert run.matching_preflight('cpu','a'*64)['path'] == str(path)
```

</details>
