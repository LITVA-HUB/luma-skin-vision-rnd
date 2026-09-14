"""Matched HR sweep; independent new output heads, frozen AS unit controls."""
import argparse
import os
import shutil
import time

import numpy as np
import torch
from chromaseed_architecture_scale import BATCH, HORIZON, RATES, SEEDS, SLOTS, VARIANTS, capacity
from chromaseed_architecture_scale_run import OUT as AS_OUT
from chromaseed_architecture_scale_run import RUN as AS_RUN
from chromaseed_architecture_scale_run import bank_path as as_bank_path
from chromaseed_architecture_scale_run import save
from chromaseed_gated import flatten, unpack
from chromaseed_head_range import predict_torch
from chromaseed_head_range_fit import fit
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import ROOT, context
from chromaseed_patch8_numpy import choose
from chromaseed_refine_train import setup
from chromaseed_widen_run import check_map, load_data
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json

RUN = ROOT / 'experiments/runs/chromaseed_head_range_v1'
OUT = ROOT / 'docs/benchmarks/chromaseed_head_range_v1'
TIMES = (128, 512, 2048)
MODES = ('unit', 'wide', 'linear')


def pair_key(variant, mode):
    if variant not in VARIANTS or mode not in MODES:
        raise ValueError('Unregistered architecture/head pair')
    return variant + '__' + mode


def bank_path(role, variant, mode, fold=None):
    if mode == 'unit':
        return as_bank_path(role, variant, fold)
    return RUN / ('final' if fold is None else 'inner') / role / pair_key(variant, mode) / ('bank' if fold is None else f'fold{fold}')


def freeze():
    seal_path = AS_OUT / 'verification.json'
    assert sha(seal_path) == 'ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9'
    parent = js(seal_path)
    sources = {**parent['sources'], **parent['postprocess_sources']}
    inputs = {**parent['inputs'], **parent['artifact_sha256']}
    inputs[seal_path.relative_to(ROOT).as_posix()] = sha(seal_path)
    probe = js(RUN / 'preflight.json')
    assert probe['passed'] and probe['exact_unit_payloads'] == 84
    assert probe['source_sha256'] == sha(ROOT / 'scripts/chromaseed_head_range_preflight.py')
    assert probe['preflight_protocol_sha256'] == sha(RUN / 'preflight_protocol.json')
    for name in ['scripts/chromaseed_head_range.py', 'scripts/chromaseed_head_range_fit.py',
                 'scripts/chromaseed_head_range_preflight.py', 'scripts/chromaseed_head_range_run.py',
                 'tests/test_chromaseed_head_range.py', 'tests/test_chromaseed_head_range_fit.py',
                 'tests/test_chromaseed_head_range_run.py', 'docs/research/chromaseed_head_range_v1_protocol.md']:
        sources[name] = sha(ROOT / name)
    for path in [RUN/'preflight.json', RUN/'preflight_protocol.json',
                 ROOT/'docs/benchmarks/chromaseed_head_range_cpu_probe/verification.json']:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    check_map({**sources, **inputs})
    assert all(os.environ.get(k) == '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
    value = dict(sources=sources, input_sha256=inputs, cache_sha256=CACHE_HASH,
                 variants=list(VARIANTS), modes=list(MODES), slots=[list(s) for s in SLOTS],
                 checkpoints=list(TIMES), horizon=HORIZON, batch_size=BATCH,
                 new_inner_banks=126, new_final_banks=42, new_trajectories=1008,
                 candidates=384, choices=87, final_records=207,
                 torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(),
                 evidence='historically reused TRAIN-only native Lab exploratory roles')
    path = RUN/'source_lock.json'
    if path.exists():
        assert js(path) == value, 'Frozen HR code/configuration changed'
    else:
        assert shutil.disk_usage(RUN).free > 40_000_000_000
        write_json(path,value)
    return sha(path)


def verified_bank(path):
    receipt = js(path/'receipt.json')
    for name, digest in receipt['files'].items():
        assert sha(path/name) == digest, path/name
    return receipt


def train_bank(data, source, role, variant, mode, completed, fold=None, steps=2048):
    ix, query, warm, parents = context(data, role, fold)
    path = bank_path(role,variant,mode,fold)
    selector = None if fold is not None else sha(RUN/'selections.json')
    if (path/'receipt.json').exists():
        old = verified_bank(path)
        assert old['source_lock_sha256'] == source and old['selection_sha256'] == selector
        assert old['warm_parent_sha256'] == parents and old['head_mode'] == mode
        print('HR REUSE',role,variant,mode,fold,flush=True)
        return
    old = js(as_bank_path(role,variant,fold)/'receipt.json')
    assert old['warm_parent_sha256'] == parents
    rows = nz(as_bank_path(role,variant,fold)/'rows.npz')
    np.testing.assert_array_equal(rows['fit_rows'],ix)
    if fold is not None:
        np.testing.assert_array_equal(rows['query_rows'],query)
    assert shutil.disk_usage(RUN).free > 15_000_000_000
    started = time.perf_counter()
    base_progress = dict(status='training',pid=os.getpid(),stage='inner' if fold is not None else 'final',
                         role=role,variant=variant,head_mode=mode,fold=fold,target_steps=steps,
                         completed_banks=completed,total_banks=168,parameters=capacity(variant))
    write_json(RUN/'progress.json',dict(base_progress,step=0,seconds=0))
    print('HR START',role,variant,mode,fold,steps,flush=True)

    def progress(item):
        write_json(RUN/'progress.json',dict(base_progress,**item))
        print('HR TRAIN',role,variant,mode,fold,item['step'],round(item['seconds'],2),flush=True)

    checkpoints = TIMES if fold is not None else (steps,)
    models, info = fit(data['color'][ix],data['tokens'][ix],data['target'][ix],
                       weights_for(data['patient'][ix],data['site'][ix]),warm,variant,mode,
                       steps,checkpoints,'cuda','cuda_graph',progress)
    files = {}

    def write(name,payload):
        save(path/name,payload)
        files[name] = sha(path/name)

    write('warm_models.npz',flatten({str(i):m for i,m in enumerate(warm)}))
    for step, ms in models.items():
        write(f'models_{step}.npz',flatten({str(i):m for i,m in enumerate(ms)}))
        if fold is not None:
            pred = np.stack([predict_torch(m,data['color'][query],data['tokens'][query]) for m in ms])
            write(f'oof_{step}.npz',dict(row_indices=query,predictions=pred))
    write('rows.npz',dict(fit_rows=ix,query_rows=query if fold is not None else np.array([],np.int64)))
    info.update(source_lock_sha256=source,selection_sha256=selector,role=role,fold=fold,
                warm_parent_sha256=parents,files=files,write_and_prediction_inclusive_seconds=time.perf_counter()-started)
    write_json(path/'receipt.json',info)
    print('HR SAVED',role,variant,mode,fold,flush=True)


def oof(role,variant,mode,rate,step):
    parts=[]
    for fold in range(3):
        path=bank_path(role,variant,mode,fold)
        receipt=js(path/'receipt.json')
        name=f'oof_{step}.npz'
        assert sha(path/name)==receipt['files'][name]
        parts.append(nz(path/name))
    indices=np.concatenate([p['row_indices'] for p in parts])
    order=np.argsort(indices)
    pred=np.concatenate([p['predictions'] for p in parts],1)[:,order]
    return indices[order],pred[[2*i+RATES.index(rate) for i in range(3)]]


def policies(candidates):
    per_pair={pair_key(v,m):choose([c for c in candidates if c['variant']==pair_key(v,m)]) for v in VARIANTS for m in MODES}
    per_architecture={v:choose([c for c in candidates if c.get('architecture')==v]) for v in VARIANTS}
    return dict(per_pair=per_pair,per_architecture=per_architecture,overall=choose(candidates))


def select(data,source):
    previous=js(AS_RUN/'selections.json')
    value=dict(source_lock_sha256=source,roles={})
    for role,(mask,_) in roles(data['patient'],data['device']).items():
        ix=np.flatnonzero(mask)
        candidates=[]
        for variant in VARIANTS:
            for mode in MODES:
                for rate in RATES:
                    for step in TIMES:
                        rows,pred=oof(role,variant,mode,rate,step)
                        np.testing.assert_array_equal(rows,ix)
                        mm=[metrics(p,data['target'][ix],data['patient'][ix],data['site'][ix]) for p in pred]
                        candidates.append(dict(variant=pair_key(variant,mode),architecture=variant,head_mode=mode,
                                               step=step,lr=rate,parameters=capacity(variant),numeric_bytes=4*capacity(variant)+458,
                                               clean=float(np.mean([m['person_mean'] for m in mm])),
                                               p90=float(np.mean([m['p90'] for m in mm])),seed_metrics=mm))
        for kind in ['np','we']:
            candidates.append(next(c for c in previous['roles'][role]['candidates'] if c['variant']==kind))
        assert len(candidates)==128
        chosen=policies(candidates)
        for variant in VARIANTS:
            unit=chosen['per_pair'][pair_key(variant,'unit')]
            prior=previous['roles'][role]['policies'][variant]
            assert (unit['step'],unit['lr'],unit['clean'])==(prior['step'],prior['lr'],prior['clean'])
        value['roles'][role]=dict(candidates=candidates,policies=chosen)
    path=RUN/'selections.json'
    if path.exists():
        assert js(path)==value
    else:
        write_json(path,value)
    print('HR SELECTION FROZEN',sha(path),flush=True)
    return value


def evaluate(data,source,selection):
    previous=js(AS_RUN/'results.json')
    records=[]
    for role,(_,mask) in roles(data['patient'],data['device']).items():
        rows=np.flatnonzero(mask)
        overall=selection['roles'][role]['policies']['overall']['variant']
        for variant in VARIANTS:
            for mode in ['wide','linear']:
                pair=pair_key(variant,mode)
                c=selection['roles'][role]['policies']['per_pair'][pair]
                path=bank_path(role,variant,mode)/f"models_{c['step']}.npz"
                bank=nz(path)
                for si,seed in enumerate(SEEDS):
                    slot=2*si+RATES.index(c['lr'])
                    model=unpack(bank,str(slot))
                    dest=RUN/'models'/role/f'{pair}_s{seed}.npz'
                    save(dest,model)
                    passes=predict_torch(model,data['color'][rows],data['tokens'][rows],all_passes=True)
                    pred=passes[:,-1]
                    out=RUN/'evaluated'/role/f'{pair}_s{seed}.npz'
                    save(out,dict(row_indices=rows,predictions=pred,passes=passes))
                    records.append(dict(role=role,variant=pair,architecture=variant,head_mode=mode,seed=seed,
                                        step=c['step'],lr=c['lr'],source_path=path.relative_to(ROOT).as_posix(),
                                        source_sha256=sha(path),slot=slot,model=dest.relative_to(ROOT).as_posix(),
                                        model_sha256=sha(dest),output=out.relative_to(ROOT).as_posix(),output_sha256=sha(out),
                                        parameters=capacity(variant),numeric_bytes=sum(v.nbytes for v in model.values() if v.dtype.kind in 'biufc'),
                                        metrics=metrics(pred,data['target'][rows],data['patient'][rows],data['site'][rows]),overall=overall==pair))
                print('HR EVALUATED',role,pair,flush=True)
        for old in [r for r in previous['records'] if r['role']==role]:
            variant=old['variant']
            pair=pair_key(variant,'unit') if variant in VARIANTS else variant
            records.append(dict(role=role,variant=pair,architecture=variant,head_mode='unit' if variant in VARIANTS else 'control',
                                seed=old['seed'],step=old['step'],lr=old['lr'],inherited_as_record=old,
                                metrics=old['metrics'],overall=overall==pair))
    assert len(records)==207 and sum(r['overall'] for r in records)==9
    value=dict(source_lock_sha256=source,selection_sha256=sha(RUN/'selections.json'),records=records)
    write_json(RUN/'results.json',value)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    setup('cuda')
    assert not (OUT/'verification.json').exists(), 'Sealed HR writers cannot rerun'
    source=freeze()
    if args.freeze_only:
        print('HR FROZEN',source,flush=True)
        return
    if (RUN/'results.json').exists():
        raise RuntimeError('Primary complete; verify results instead of rerunning')
    data=load_data()
    started=time.perf_counter()
    write_json(RUN/'job.json',dict(status='running',pid=os.getpid(),source_lock_sha256=source))
    completed=0
    try:
        for role in roles(data['patient'],data['device']):
            for variant in VARIANTS:
                for mode in ['wide','linear']:
                    for fold in range(3):
                        train_bank(data,source,role,variant,mode,completed,fold)
                        completed+=1
        selection=select(data,source)
        for role in selection['roles']:
            for variant in VARIANTS:
                for mode in ['wide','linear']:
                    steps=selection['roles'][role]['policies']['per_pair'][pair_key(variant,mode)]['step']
                    train_bank(data,source,role,variant,mode,completed,steps=steps)
                    completed+=1
        write_json(RUN/'progress.json',dict(status='evaluating',pid=os.getpid(),completed_banks=completed,total_banks=168))
        evaluate(data,source,selection)
    except BaseException as exc:
        write_json(RUN/'job.json',dict(status='failed',pid=os.getpid(),error=type(exc).__name__+': '+str(exc),completed_banks=completed))
        raise
    write_json(RUN/'job.json',dict(status='complete',pid=os.getpid(),source_lock_sha256=source,
                                  results_sha256=sha(RUN/'results.json'),seconds=time.perf_counter()-started))
    write_json(RUN/'progress.json',dict(status='complete',completed_banks=168,total_banks=168))
    print('HR PRIMARY COMPLETE',sha(RUN/'results.json'),flush=True)


if __name__=='__main__':
    main()
