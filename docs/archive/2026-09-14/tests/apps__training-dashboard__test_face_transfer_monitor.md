# `apps/training-dashboard/test_face_transfer_monitor.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../apps/training-dashboard/test_face_transfer_monitor.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `9b857203e9973bd1736a70d037075cb5d1926b2a75be1748d75d3ed18185bfa1`. Строк: **241**.

## Зависимости

```python
import hashlib
import json
import os
import sys
from pathlib import Path
import pytest
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 11](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L11)

```python
ARMS = ('lapa_only','lapa_celeba')
```

[Строка 12](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L12)

```python
SEEDS = (17,29,43)
```

[Строка 13](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L13)

```python
COUNTS = {'lapa':{'train':15914,'validation':1692,'test':2000},
          'celeba':{'train':24112,'validation':2992,'test':2822}}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `write` | FunctionDef | См. реализацию | [L17](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L17) |
| `fixture` | FunctionDef | См. реализацию | [L24](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L24) |
| `live` | FunctionDef | См. реализацию | [L41](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L41) |
| `point` | FunctionDef | См. реализацию | [L52](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L52) |
| `save_run` | FunctionDef | См. реализацию | [L57](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L57) |
| `test_real_preparation_is_waiting_with_exact_sources_and_no_fake_rate` | FunctionDef | См. реализацию | [L68](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L68) |
| `test_parent_queue_requires_source_matched_seal_and_dead_worker` | FunctionDef | См. реализацию | [L80](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L80) |
| `test_untrusted_liveness_hides_rate_and_eta` | FunctionDef | См. реализацию | [L97](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L97) |
| `test_counter_never_substitutes_for_saved_trajectory` | FunctionDef | См. реализацию | [L108](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L108) |
| `test_inconsistent_or_malformed_progress_cannot_show_training_speed` | FunctionDef | См. реализацию | [L123](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L123) |
| `test_only_source_complete_validation_points_appear_on_chart` | FunctionDef | См. реализацию | [L145](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L145) |
| `test_immutable_running_job_cannot_override_complete_receipts` | FunctionDef | См. реализацию | [L160](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L160) |
| `test_foreign_completion_and_fake_seal_are_not_quality_evidence` | FunctionDef | См. реализацию | [L175](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L175) |
| `test_missing_registration_hides_card_and_broken_registration_is_safe` | FunctionDef | См. реализацию | [L185](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L185) |
| `test_malformed_nested_preflight_never_breaks_the_entire_api` | FunctionDef | См. реализацию | [L195](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L195) |
| `test_post_training_stages_and_verified_quality_follow_artifact_links` | FunctionDef | См. реализацию | [L206](../../../../apps/training-dashboard/test_face_transfer_monitor.py#L206) |

## Все тестовые определения (11)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_real_preparation_is_waiting_with_exact_sources_and_no_fake_rate` · L68

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_real_preparation_is_waiting_with_exact_sources_and_no_fake_rate(tmp_path):
    from face_transfer_monitor import snapshot
    _,_,_,options = fixture(tmp_path)
    row = snapshot(**options)
    assert row['state']=='waiting_hr' and row['cpu']['passed'] and row['cpu']['cases']==6
    assert row['completed_runs']==0 and row['total_runs']==6 and len(row['runs'])==6
    assert row['total_updates']==35856 and row['observed_updates']==0
    assert row['counts']==COUNTS and row['training_images']==40026
    assert row['images_per_second'] is row['estimated_remaining_seconds'] is None
    assert row['quality'] is None and not row['verified']
```

</details>

### `test_parent_queue_requires_source_matched_seal_and_dead_worker` · L80

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_parent_queue_requires_source_matched_seal_and_dead_worker(tmp_path):
    from face_transfer_monitor import snapshot
    root,registration,_,options = fixture(tmp_path)
    for kind,name in [('hr','chromaseed_head_range_v1'),('p3','chromaseed_palette_transfer_v1')]:
        base=options[kind+'_root']
        source=write(base/'source_lock.json',{'bindings':{}})
        write(base/'job.json',{'status':'complete','pid':80})
        write(options['project']/f'docs/benchmarks/{name}/verification.json',{'passed':True,'source_lock_sha256':source})
        assert snapshot(**options)['state']==('waiting_p3' if kind=='hr' else 'needs_gpu_check')
    cpu=json.loads((root/'preflights/cpu_good/result.json').read_text())
    write(root/'preflights/cuda_good/result.json',{**cpu,'device':'cuda','batch_size':32,'registration_sha256':registration})
    assert snapshot(**options)['state']=='ready'
    options['probe']=lambda p:True
    assert snapshot(**options)['state']!='ready'
```

</details>

### `test_untrusted_liveness_hides_rate_and_eta` · L97

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('alive,now,state', [(False, 1100, 'interrupted'), (None, 1100, 'unknown'), (True, 1400, 'stale')])
def test_untrusted_liveness_hides_rate_and_eta(tmp_path,alive,now,state):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options)
    options.update(probe=lambda p: alive if p==42 else False,now=now)
    row=snapshot(**options)
    assert row['state']==state
    assert row['images_per_second'] is row['estimated_remaining_seconds'] is None
    assert not any(r['status']=='running' for r in row['runs'])
```

</details>

### `test_counter_never_substitutes_for_saved_trajectory` · L108

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_counter_never_substitutes_for_saved_trajectory(tmp_path):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options,step=5976)
    row=snapshot(**options)
    assert row['completed_runs']==0 and row['observed_updates']==5976
    assert row['state']=='training' and row['images_per_second']==1912.32
    save_run(root,spec,0)
    live(root,registration,spec,options,index=1,step=498)
    row=snapshot(**options)
    assert row['completed_runs']==1 and row['observed_updates']==6474
    assert row['runs'][0]['status']=='completed' and row['runs'][1]['status']=='running'
```

</details>

### `test_inconsistent_or_malformed_progress_cannot_show_training_speed` · L123

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('defect', ['pid', 'count', 'steps', 'images', 'nan', 'structure'])
def test_inconsistent_or_malformed_progress_cannot_show_training_speed(tmp_path,defect):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    p=live(root,registration,spec,options)
    if defect=='pid':
        p['pid']=99
    elif defect=='count':
        p['completed_updates']=999999
    elif defect=='steps':
        p['step']=5977
    elif defect=='images':
        p['images_seen']=9
    elif defect=='nan':
        p['elapsed_seconds']=float('nan')
    else:
        p=[]
    write(root/'progress.json',p)
    row=snapshot(**options)
    assert row['state'] in ('unknown','degraded')
    assert row['images_per_second'] is row['estimated_remaining_seconds'] is None
```

</details>

### `test_only_source_complete_validation_points_appear_on_chart` · L145

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_only_source_complete_validation_points_appear_on_chart(tmp_path):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options)
    r=spec['trajectories'][0]
    p=point(root,r,498)
    write(root/'trajectories'/r['id']/'step_0498.json',p)
    row=snapshot(**options)
    assert row['runs'][0]['history']==[{'step':498,'lapa_iou':.9,'celeba_iou':.8,'macro_iou':pytest.approx(.85)}]
    del p['validation']['celeba']
    write(root/'trajectories'/r['id']/'step_0498.json',p)
    row=snapshot(**options)
    assert row['runs'][0]['history']==[] and row['warnings']
```

</details>

### `test_immutable_running_job_cannot_override_complete_receipts` · L160

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_immutable_running_job_cannot_override_complete_receipts(tmp_path):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options,index=5,step=5976)
    receipts=[save_run(root,spec,i) for i in range(6)]
    selection=write(root/'selection.json',{'registration_sha256':registration,'choices':[{}]*18})
    write(root/'completion.json',dict(status='complete',pid=42,registration_sha256=registration,
          successful_updates=35856,selection_sha256=selection,trajectories=receipts))
    options['probe']=lambda p:False
    row=snapshot(**options)
    assert row['state']=='waiting_test' and row['completed_runs']==6
    assert row['observed_updates']==35856 and not row['verified']
    assert row['images_per_second'] is row['estimated_remaining_seconds'] is None
```

</details>

### `test_foreign_completion_and_fake_seal_are_not_quality_evidence` · L175

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_foreign_completion_and_fake_seal_are_not_quality_evidence(tmp_path):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options)
    write(root/'completion.json',dict(status='complete',pid=42,registration_sha256='foreign',successful_updates=35856))
    write(root/'verification.json',{'passed':True,'artifacts':{}})
    row=snapshot(**options)
    assert not row['verified'] and row['quality'] is None and row['state']=='degraded'
```

</details>

### `test_missing_registration_hides_card_and_broken_registration_is_safe` · L185

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_missing_registration_hides_card_and_broken_registration_is_safe(tmp_path):
    from face_transfer_monitor import snapshot
    _,_,_,options=fixture(tmp_path)
    options['root']=tmp_path/'empty'
    assert snapshot(**options) is None
    write(options['root']/'registration.json',{'recipe':None})
    row=snapshot(**options)
    assert row['state']=='degraded' and row['warnings']
```

</details>

### `test_malformed_nested_preflight_never_breaks_the_entire_api` · L195

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_malformed_nested_preflight_never_breaks_the_entire_api(tmp_path):
    from face_transfer_monitor import snapshot
    root,_,_,options=fixture(tmp_path)
    path=root/'preflights/cpu_good/result.json'
    value=json.loads(path.read_text())
    value['cases'][0]['arm']=[]
    write(path,value)
    row=snapshot(**options)
    assert not row['cpu']['passed']
```

</details>

### `test_post_training_stages_and_verified_quality_follow_artifact_links` · L206

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_post_training_stages_and_verified_quality_follow_artifact_links(tmp_path):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options,index=5,step=5976)
    receipts=[save_run(root,spec,i) for i in range(6)]
    selection=write(root/'selection.json',{'registration_sha256':registration,'choices':[{}]*18})
    write(root/'completion.json',dict(status='complete',pid=42,registration_sha256=registration,
          successful_updates=35856,selection_sha256=selection,trajectories=receipts))
    options['probe']=lambda p:p==43
    test_protocol=write(root/'test_protocol.json',{'pid':43,'selection_sha256':selection,'registration_sha256':registration})
    assert snapshot(**options)['state']=='evaluating'
    test_result=write(root/'test_results.json',{'status':'complete','pid':43,'test_protocol_sha256':test_protocol})
    options['probe']=lambda p:False
    assert snapshot(**options)['state']=='waiting_audit'
    audit=write(root/'test_audit.json',{'status':'complete','pid':44,'passed':True,'artifacts':{str(root/'test_results.json'):test_result}})
    assert snapshot(**options)['state']=='waiting_runtime'
    runtime_protocol=write(root/'runtime_protocol.json',{'pid':45,'test_audit_sha256':audit})
    options['probe']=lambda p:p==45
    assert snapshot(**options)['state']=='benchmarking'
    options['probe']=lambda p:False
    write(root/'runtime.json',{'status':'complete','pid':45,'protocol_sha256':runtime_protocol})
    assert snapshot(**options)['state']=='waiting_verification'
    write(root/'verification.json',{'passed':True,'artifacts':{}})
    assert snapshot(**options)['quality'] is None
    chosen={'arm':'lapa_celeba','seed':17,'budget':5976}
    write(root/'summary.json',{'overall_validation_choice':chosen,'all_choices':[{**chosen,'selected_step':1494,
          'sources':{s:{'metrics':{'mean_image_iou':.9,'images':COUNTS[s]['test']}} for s in COUNTS}}]})
    names=['registration.json','completion.json','selection.json','test_protocol.json','test_results.json',
           'test_audit.json','runtime_protocol.json','runtime.json','summary.json']
    bindings={str(root/name):hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    write(root/'verification.json',{'passed':True,'artifacts':bindings})
    row=snapshot(**options)
    assert row['state']=='verified' and row['verified'] and row['quality']['iou']=={'lapa':.9,'celeba':.9}
    assert row['images_per_second'] is row['estimated_remaining_seconds'] is None
    write(root/'summary.json',{'tampered':True})
    assert not snapshot(**options)['verified']
```

</details>
