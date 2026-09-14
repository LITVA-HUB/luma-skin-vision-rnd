import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))

ARMS = ('lapa_only','lapa_celeba')
SEEDS = (17,29,43)
COUNTS = {'lapa':{'train':15914,'validation':1692,'test':2000},
          'celeba':{'train':24112,'validation':2992,'test':2822}}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    os.utime(path,(1000,1000))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    root = tmp_path/'seg2'
    runs = [dict(id=f'{arm}_s{seed}',arm=arm,seed=seed,steps=5976,image_presentations=191232,
                 source_presentations={'lapa':191232 if arm==ARMS[0] else 95616,'celeba':0 if arm==ARMS[0] else 95616})
            for arm in ARMS for seed in SEEDS]
    spec = dict(trajectories=runs,budgets=[1494,2988,5976],validation_steps=list(range(0,5977,498)),
                validation_interval=498,batch_size=32,parameters=4416673,counts=COUNTS,
                initial_sha256='a'*64,new_updates=35856,new_image_presentations=1147392)
    registration = write(root/'registration.json',{'recipe':spec,'bindings':{}})
    cases = [dict(arm=r['arm'],seed=r['seed'],parameters=4416673,weights_changed=True,
                  steps=[dict(successful_update=True,loss=.1,gradient_norm=.3) for _ in range(2)]) for r in runs]
    write(root/'preflights/cpu_good/result.json',dict(status='passed',device='cpu',pid=90,
          registration_sha256=registration,successful_updates=12,anchor_pairs_exact=True,batch_size=2,cases=cases))
    options = dict(root=root,project=tmp_path,hr_root=tmp_path/'hr',p3_root=tmp_path/'p3',now=1100,probe=lambda p:False)
    return root,registration,spec,options


def live(root,registration,spec,options,index=0,step=498):
    write(root/'job.json',dict(status='running',pid=42,registration_sha256=registration))
    progress = dict(status='training',pid=42,trajectory=spec['trajectories'][index]['id'],step=step,total_steps=5976,
                    completed_updates=index*5976+step,total_updates=35856,elapsed_seconds=100,
                    images_seen=(index*5976+step)*32,images_per_second=(index*5976+step)*32/100,
                    estimated_remaining_seconds=100/(index*5976+step)*(35856-index*5976-step),training_loss=.1)
    write(root/'progress.json',progress)
    options['probe'] = lambda p: p==42
    return progress


def point(root,run,step):
    return dict(step=step,checkpoint={'path':str(root/'trajectories'/run['id']/f'step_{step:04d}.pt'),'sha256':'a'*64},
                validation={s:dict(images=COUNTS[s]['validation'],mean_image_iou=.9 if s=='lapa' else .8) for s in COUNTS})


def save_run(root,spec,index):
    run = spec['trajectories'][index]
    directory = root/'trajectories'/run['id']
    history = [point(root,run,s) for s in spec['validation_steps']]
    sha = write(directory/'history.json',history)
    record = dict(**run,successful_updates=5976,attempted_batches=5976,history_path=str(directory/'history.json'),
                  history_sha256=sha,elapsed_seconds=200)
    write(directory/'receipt.json',record)
    return record


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


@pytest.mark.parametrize('alive,now,state',[(False,1100,'interrupted'),(None,1100,'unknown'),(True,1400,'stale')])
def test_untrusted_liveness_hides_rate_and_eta(tmp_path,alive,now,state):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options)
    options.update(probe=lambda p: alive if p==42 else False,now=now)
    row=snapshot(**options)
    assert row['state']==state
    assert row['images_per_second'] is row['estimated_remaining_seconds'] is None
    assert not any(r['status']=='running' for r in row['runs'])


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


@pytest.mark.parametrize('defect',['pid','count','steps','images','nan','structure'])
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


def test_foreign_completion_and_fake_seal_are_not_quality_evidence(tmp_path):
    from face_transfer_monitor import snapshot
    root,registration,spec,options=fixture(tmp_path)
    live(root,registration,spec,options)
    write(root/'completion.json',dict(status='complete',pid=42,registration_sha256='foreign',successful_updates=35856))
    write(root/'verification.json',{'passed':True,'artifacts':{}})
    row=snapshot(**options)
    assert not row['verified'] and row['quality'] is None and row['state']=='degraded'


def test_missing_registration_hides_card_and_broken_registration_is_safe(tmp_path):
    from face_transfer_monitor import snapshot
    _,_,_,options=fixture(tmp_path)
    options['root']=tmp_path/'empty'
    assert snapshot(**options) is None
    write(options['root']/'registration.json',{'recipe':None})
    row=snapshot(**options)
    assert row['state']=='degraded' and row['warnings']


def test_malformed_nested_preflight_never_breaks_the_entire_api(tmp_path):
    from face_transfer_monitor import snapshot
    root,_,_,options=fixture(tmp_path)
    path=root/'preflights/cpu_good/result.json'
    value=json.loads(path.read_text())
    value['cases'][0]['arm']=[]
    write(path,value)
    row=snapshot(**options)
    assert not row['cpu']['passed']


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
