"""Read-only Seg2 monitor: JSON metadata and process state, never ML/data-array imports."""
import hashlib
import json
import math
import time
from pathlib import Path

from monitor import process_alive

PROJECT = Path(__file__).resolve().parents[2]
ARMS = ('lapa_only','lapa_celeba')
SEEDS = (17,29,43)
STEPS, BATCH, TOTAL = 5976,32,35856
COUNTS = {'lapa':{'train':15914,'validation':1692,'test':2000},
          'celeba':{'train':24112,'validation':2992,'test':2822}}


def finite(value,minimum=0):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=minimum


def integer(value,minimum=0,maximum=TOTAL):
    return isinstance(value,int) and not isinstance(value,bool) and minimum<=value<=maximum


def snapshot(root=Path('D:/Luma-RnD/skin_face_transfer_v1'),project=PROJECT,
             hr_root=Path('D:/Luma-RnD/chromaseed_head_range_v1'),
             p3_root=Path('D:/Luma-RnD/chromaseed_palette_transfer_v1'),now=None,probe=process_alive):
    root,project = Path(root),Path(project)
    now = time.time() if now is None else now
    warnings,hashes,mtimes,cache = [],{},{},{}

    def read(path,kind=dict):
        path=Path(path)
        key=path,kind
        if key in cache:
            return cache[key]
        try:
            raw=path.read_bytes()
            def invalid(value):
                raise ValueError(f'Nonfinite JSON: {value}')
            value=json.loads(raw,parse_constant=invalid)
            if not isinstance(value,kind):
                raise ValueError('Wrong JSON structure')
            hashes[path]=hashlib.sha256(raw).hexdigest()
            mtimes[path]=path.stat().st_mtime
        except FileNotFoundError:
            value=kind()
        except (OSError,ValueError):
            warnings.append(f'Не удалось прочитать запись {path.name}; ожидаем корректные данные.')
            value=kind()
        cache[key]=value
        return value

    registration=read(root/'registration.json')
    if not registration and not warnings:
        return None
    spec=registration.get('recipe')
    valid_spec=isinstance(spec,dict)
    expected=[(a,s,f'{a}_s{s}') for a in ARMS for s in SEEDS]
    if valid_spec:
        runs=spec.get('trajectories',[])
        valid_spec=(isinstance(runs,list) and len(runs)==6 and all(isinstance(r,dict) for r in runs)
                    and [(r.get('arm'),r.get('seed'),r.get('id')) for r in runs]==expected
                    and all(r.get('steps')==STEPS for r in runs)
                    and spec.get('new_updates')==TOTAL and spec.get('batch_size')==BATCH
                    and spec.get('parameters')==4416673 and spec.get('counts')==COUNTS
                    and spec.get('budgets')==[1494,2988,5976]
                    and spec.get('validation_steps')==list(range(0,5977,498)))
    if not valid_spec:
        warnings.append('Регистрация Seg2 повреждена или не соответствует поддерживаемому плану.')
        spec={'trajectories':[],'budgets':[],'counts':{},'validation_steps':[]}
    reg_sha=hashes.get(root/'registration.json') if valid_spec else None

    def preflight(device):
        accepted=[]
        for path in sorted((root/'preflights').glob(f'{device}_*/result.json')):
            row=read(path)
            cases=row.get('cases',[])
            good=(reg_sha and row.get('status')=='passed' and row.get('registration_sha256')==reg_sha
                  and row.get('device')==device and row.get('successful_updates')==12
                  and row.get('anchor_pairs_exact') is True and row.get('batch_size')==(2 if device=='cpu' else 32)
                  and isinstance(cases,list) and len(cases)==6 and all(isinstance(c,dict) for c in cases)
                  and all(isinstance(c.get('arm'),str) and isinstance(c.get('seed'),int) for c in cases)
                  and {(c.get('arm'),c.get('seed')) for c in cases}=={(a,s) for a,s,_ in expected}
                  and probe(row.get('pid')) is False)
            if good:
                accepted.append(row)
        return dict(passed=bool(accepted),cases=6 if accepted else 0,updates=12 if accepted else 0)

    cpu,gpu=preflight('cpu'),preflight('cuda')
    parents={}
    for name,directory in [('hr',Path(hr_root)),('p3',Path(p3_root))]:
        series='chromaseed_head_range_v1' if name=='hr' else 'chromaseed_palette_transfer_v1'
        read(directory/'source_lock.json')
        job=read(directory/'job.json')
        seal=read(project/f'docs/benchmarks/{series}/verification.json')
        source=hashes.get(directory/'source_lock.json')
        parents[name]=bool(source and seal.get('passed') is True and seal.get('source_lock_sha256')==source
                           and job.get('status')=='complete' and probe(job.get('pid')) is False)
    state=('preparing' if not cpu['passed'] else 'waiting_hr' if not parents['hr'] else
           'waiting_p3' if not parents['p3'] else 'needs_gpu_check' if not gpu['passed'] else 'ready')
    job,progress=read(root/'job.json'),read(root/'progress.json')
    failure,completion=read(root/'failure.json'),read(root/'completion.json')
    selected=read(root/'selection.json')
    job_ok=bool(reg_sha and job.get('registration_sha256')==reg_sha and integer(job.get('pid'),1,2**32))
    entries,receipts=[],[]
    zero=read(root/'baseline/validation.json')

    def point(row,step):
        metrics=row.get('validation')
        checkpoint=row.get('checkpoint')
        if (row.get('step')!=step or not isinstance(metrics,dict) or set(metrics)!=set(COUNTS)
                or not isinstance(checkpoint,dict) or not isinstance(checkpoint.get('sha256'),str)
                or len(checkpoint['sha256'])!=64):
            return None
        values={}
        for source in COUNTS:
            m=metrics[source]
            if (not isinstance(m,dict) or m.get('images')!=COUNTS[source]['validation']
                    or not finite(m.get('mean_image_iou')) or m['mean_image_iou']>1):
                return None
            values[source]=m['mean_image_iou']
        return dict(step=step,lapa_iou=values['lapa'],celeba_iou=values['celeba'],macro_iou=sum(values.values())/2)

    for run in spec['trajectories']:
        directory=root/'trajectories'/run['id']
        receipt=read(directory/'receipt.json')
        history=read(directory/'history.json',list) if receipt else []
        valid=(job_ok and bool(receipt) and all(receipt.get(k)==v for k,v in run.items())
               and receipt.get('successful_updates')==STEPS and receipt.get('attempted_batches')==STEPS
               and receipt.get('history_path')==str(directory/'history.json')
               and receipt.get('history_sha256')==hashes.get(directory/'history.json')
               and len(history)==13 and all(isinstance(r,dict) for r in history)
               and [r.get('step') for r in history]==spec['validation_steps']
               and all(point(row,step) is not None for row,step in zip(history,spec['validation_steps'])))
        if receipt and not valid:
            warnings.append(f"Запуск {run['id']}: сохранение не совпадает с планом или историей.")
        trace=[]
        candidates=history if valid else ([zero] if zero else [])+[read(directory/f'step_{step:04d}.json') for step in spec['validation_steps'][1:]]
        for row in candidates:
            if not row:
                continue
            p=point(row,row.get('step')) if isinstance(row,dict) and row.get('step') in spec['validation_steps'] else None
            if p and job_ok:
                trace.append(p)
            elif not p:
                warnings.append('Одна из валидационных точек повреждена; она не показана на графике.')
        entries.append(dict(id=run['id'],arm=run['arm'],seed=run['seed'],target_steps=STEPS,
                            step=STEPS if valid else None,status='completed' if valid else 'queued',
                            history=trace,seconds=receipt.get('elapsed_seconds') if valid else None))
        receipts.append(receipt if valid else None)
    completed=sum(r is not None for r in receipts)
    saved_updates=completed*STEPS
    current=None
    observed=saved_updates
    rate=eta=None
    if job:
        live=probe(job.get('pid')) if job_ok else None
        if not job_ok:
            state='degraded'
            warnings.append('Версия активного запуска Seg2 не подтверждена.')
        elif live is False:
            state='interrupted'
        elif live is None:
            state='unknown'
        elif not progress:
            state='initializing'
        elif progress.get('pid')!=job.get('pid'):
            state='unknown'
        else:
            index=next((i for i,r in enumerate(entries) if r['id']==progress.get('trajectory')),None)
            step=progress.get('step')
            valid=(index is not None and integer(step,0,STEPS)
                   and progress.get('total_steps')==STEPS and progress.get('total_updates')==TOTAL
                   and progress.get('completed_updates')==index*STEPS+step
                   and progress.get('images_seen')==(index*STEPS+step)*BATCH
                   and all(receipts[i] is not None for i in range(index)))
            if valid:
                observed=max(saved_updates,progress['completed_updates'])
                current={**entries[index],'step':step,'loss':progress.get('training_loss') if finite(progress.get('training_loss')) else None}
                entries[index]['step']=STEPS if receipts[index] else step
                if now-mtimes.get(root/'progress.json',0)>180 or mtimes.get(root/'progress.json',0)>now+5:
                    state='stale'
                elif progress.get('status')=='training':
                    state='training'
                    if not receipts[index]:
                        entries[index]['status']='running'
                    elapsed=progress.get('elapsed_seconds')
                    if finite(elapsed,1e-9) and observed>0:
                        rate=observed*BATCH/elapsed
                        eta=elapsed/observed*(TOTAL-observed)
                else:
                    state='finishing'
            else:
                state='degraded'
                warnings.append('Текущие шаги Seg2 не согласованы с сохранёнными запусками.')
    complete_ok=(job_ok and bool(completion) and completion.get('status')=='complete'
                 and completion.get('pid')==job.get('pid') and completion.get('registration_sha256')==reg_sha
                 and completion.get('successful_updates')==TOTAL and completed==6
                 and completion.get('trajectories')==receipts
                 and completion.get('selection_sha256')==hashes.get(root/'selection.json')
                 and selected.get('registration_sha256')==reg_sha and isinstance(selected.get('choices'),list)
                 and len(selected['choices'])==18)
    if completion and not complete_ok:
        state='degraded'
        warnings.append('Завершение Seg2 не подтверждено всеми шестью запусками и выбором настроек.')
    if complete_ok:
        live=probe(completion.get('pid'))
        state='waiting_test' if live is False else 'finishing' if live is True else 'unknown'
        observed,current,rate,eta=TOTAL,None,None,None
    if failure and job_ok and not complete_ok:
        state='failed'
    verified,quality=False,None
    if complete_ok and probe(completion.get('pid')) is False:
        test_protocol,test_result=read(root/'test_protocol.json'),read(root/'test_results.json')
        test_ok=(test_protocol.get('selection_sha256')==hashes.get(root/'selection.json')
                 and test_protocol.get('registration_sha256')==reg_sha
                 and test_result.get('test_protocol_sha256')==hashes.get(root/'test_protocol.json')
                 and test_result.get('status')=='complete' and probe(test_result.get('pid')) is False)
        if test_protocol and not test_result and probe(test_protocol.get('pid')) is True:
            state='evaluating'
        elif test_ok:
            state='waiting_audit'
        audit=read(root/'test_audit.json')
        audit_bindings=audit.get('artifacts',{})
        audit_ok=(test_ok and audit.get('passed') is True and isinstance(audit_bindings,dict)
                  and audit_bindings.get(str(root/'test_results.json'))==hashes.get(root/'test_results.json')
                  and audit.get('status')=='complete' and probe(audit.get('pid')) is False)
        if audit_ok:
            state='waiting_runtime'
        runtime_protocol,runtime=read(root/'runtime_protocol.json'),read(root/'runtime.json')
        if audit_ok and runtime_protocol and not runtime and probe(runtime_protocol.get('pid')) is True:
            state='benchmarking'
        runtime_ok=(audit_ok and runtime.get('status')=='complete' and probe(runtime.get('pid')) is False
                    and runtime.get('protocol_sha256')==hashes.get(root/'runtime_protocol.json')
                    and runtime_protocol.get('test_audit_sha256')==hashes.get(root/'test_audit.json'))
        if runtime_ok:
            state='waiting_verification'
        seal=read(root/'verification.json')
        result=read(root/'summary.json') if seal else {}
        bindings=seal.get('artifacts',{})
        verified=bool(runtime_ok and seal.get('passed') is True and isinstance(bindings,dict) and result
                      and all(hashes.get(root/name) and bindings.get(str(root/name))==hashes[root/name] for name in (
                          'registration.json','completion.json','selection.json','test_protocol.json','test_results.json',
                          'test_audit.json','runtime_protocol.json','runtime.json','summary.json')))
        if verified:
            overall=result.get('overall_validation_choice',{})
            choices=result.get('all_choices',[])
            chosen=next((r for r in choices if isinstance(r,dict) and isinstance(overall,dict)
                         and all(r.get(k)==overall.get(k) for k in ('arm','seed','budget'))),None) if isinstance(choices,list) else None
            if chosen and isinstance(chosen.get('sources'),dict) and set(chosen['sources'])==set(COUNTS):
                values={}
                for source,row in chosen['sources'].items():
                    m=row.get('metrics',{}) if isinstance(row,dict) else {}
                    if finite(m.get('mean_image_iou')) and m['mean_image_iou']<=1 and m.get('images')==COUNTS[source]['test']:
                        values[source]=m['mean_image_iou']
                if len(values)==2:
                    quality=dict(arm=chosen['arm'],seed=chosen['seed'],step=chosen.get('selected_step'),iou=values)
            if quality is None:
                verified=False
                warnings.append('Итоговый выбор или метрики Seg2 требуют проверки.')
            else:
                state='verified'
    if warnings:
        if state not in ('failed','interrupted','unknown','stale'):
            state='degraded'
        rate=eta=None
        verified,quality=False,None
        for entry in entries:
            if entry['status']=='running':
                entry['status']='queued'
    if state!='training':
        rate=eta=None
    return dict(name='Luma ChromaSeed-Seg2',state=state,registration_valid=bool(reg_sha),
                parameters=4416673 if valid_spec else None,counts=spec['counts'],training_images=40026 if valid_spec else None,
                cpu=cpu,gpu=gpu,parents=parents,runs=entries,total_runs=6,completed_runs=completed,
                total_updates=TOTAL,observed_updates=observed,saved_updates=saved_updates,current=current,
                batch_size=BATCH,budgets=spec['budgets'],images_per_second=rate,estimated_remaining_seconds=eta,
                verified=verified,quality=quality,warnings=list(dict.fromkeys(warnings))[:3],
                error=failure.get('error') if state=='failed' and isinstance(failure.get('error'),str) else None,
                limitation='IoU измеряет совпадение маски кожи. Точность физического оттенка кожи этим экспериментом не подтверждается.')
