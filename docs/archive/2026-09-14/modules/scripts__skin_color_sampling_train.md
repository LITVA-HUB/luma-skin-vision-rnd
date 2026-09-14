# `scripts/skin_color_sampling_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_color_sampling_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed-budget native skin-color support allocation, TRAIN-only.

SHA-256 исходника: `5e708c79849bfd824475a3d7b6c88c00f5c614ccf7771975ee08861246fcdaf3`. Строк: **95**.

## Зависимости

```python
import os
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_color_sampling import ARMS,sampling_distribution,draw_indices
from skin_support_curve import SkinRepresentation,patient_roles
from skin_support_curve_train import predict,novelty,role_digest,bindings as prior_bindings
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_pair_train import subset,rows,write
from skin_mskcc_summary_pilot import summarize
from skin_distribution_train import score
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/skin_color_sampling_train.py#L20)

```python
OUT=ROOT/'docs/benchmarks/skin_color_sampling_v1'
```

[Строка 20](../../../../scripts/skin_color_sampling_train.py#L20)

```python
RUN=ROOT/'experiments/runs/skin_color_sampling_v1'
```

[Строка 21](../../../../scripts/skin_color_sampling_train.py#L21)

```python
PROTOCOL=ROOT/'docs/research/skin_color_sampling_protocol_v1.md'
```

[Строка 21](../../../../scripts/skin_color_sampling_train.py#L21)

```python
SEEDS=(17,29,43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `fit` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_color_sampling_train.py#L24) |
| `bindings` | FunctionDef | См. реализацию | [L77](../../../../scripts/skin_color_sampling_train.py#L77) |
| `main` | FunctionDef | См. реализацию | [L82](../../../../scripts/skin_color_sampling_train.py#L82) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L24–74</summary>

```python
def fit(data,arm,seed):
    name=f'{arm}__s{seed}';out=OUT/name;folder=RUN/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes());assert sha(folder/'final.pt')==r['checkpoint_sha256'] and sha(folder/'evaluation.npz')==r['evaluation_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    out.mkdir(parents=True,exist_ok=False);folder.mkdir(parents=True,exist_ok=False)
    a,b=patient_roles(data['patient'],data['device'],18,seed);tr,ev=subset(data,a),subset(data,b)
    assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
    q,correction,info=sampling_distribution(tr,arm)
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=SkinRepresentation('baseline');initial=digest(model.state_dict());model=model.cuda()
    ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();rgb=torch.from_numpy(tr['rgb']).cuda().float()/255
    vx=torch.from_numpy(ev['tokens']).cuda();vrgb=torch.from_numpy(ev['rgb']).cuda().float()/255
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda');importance=torch.tensor(correction,dtype=torch.float32,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
    trace=hashlib.sha256();counts=np.zeros(len(q),dtype=np.int64);history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,31):
        model.train();indices=draw_indices(q,arm,seed,epoch);trace.update(indices.tobytes());counts+=np.bincount(indices.ravel(),minlength=len(q));losses=[]
        for batch in indices:
            ix=torch.from_numpy(batch).cuda();opt.zero_grad(set_to_none=True);p,g,_=model(x[ix],rgb[ix])
            if arm=='color_ipw':loss=(((p-y[ix]).square().mean(1)+.1*torch.nn.functional.cross_entropy(g,mode[ix],reduction='none'))*importance[ix]).mean()
            else:loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=predict(model,vx,vrgb,mean,std);met=summarize(delta_e00(p,ev['target']),rows(ev))
        history.append({'round':epoch,'loss':float(np.mean(losses)),'holdout_mean':met['mean'],'holdout_patient_mean':met['patient_balanced_mean']})
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    state={'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    torch.save(state,folder/'final.pt');risk=novelty(tr,ev);scores,error,order=score(p,risk,ev)
    scores['site_balanced_mean']=float(np.mean([error[ev['site']==s].mean() for s in np.unique(ev['site'])]))
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],patient=ev['patient'],site=ev['site'],risk=risk,error=error,curve=np.cumsum(error[order])/np.arange(1,len(error)+1))
    np.savez(folder/'sampling.npz',proposal=q,correction=correction,counts=counts,site_probability=info['site_probability'],density=info['density'])
    exact_old=False
    if arm=='image':
        old=torch.load(ROOT/'experiments/runs/skin_support_curve_v1'/f'n18__baseline__s{seed}'/'final.pt',map_location='cpu',weights_only=True)
        assert digest(old['state'])==digest(state['state']);exact_old=True
    r={'arm':arm,'seed':seed,'train_people':18,'holdout_people':6,'train_images':len(x),'holdout_images':len(vx),'train_sites':info['site_count'],
        'scores':scores,'rounds':30,'optimizer_updates':930,'sampled_images':int(counts.sum()),'draw_sha256':trace.hexdigest(),'role_sha256':role_digest(a,b),
        'initial_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),'fit_seconds':seconds,'fit_peak_allocated_mib':peak,
        'checkpoint_bytes':(folder/'final.pt').stat().st_size,'checkpoint_sha256':sha(folder/'final.pt'),'evaluation_sha256':sha(folder/'evaluation.npz'),'sampling_sha256':sha(folder/'sampling.npz'),
        'effective_image_mass':float(1/np.sum(q*q)),'proposal_vs_uniform_site_min':float(info['site_probability'].min()*info['site_count']),
        'proposal_vs_uniform_site_max':float(info['site_probability'].max()*info['site_count']),'importance_min':float(correction.min()),'importance_max':float(correction.max()),
        'sampled_unique_images':int(np.sum(counts>0)),'sampled_unique_sites':len(set(tr['site'][counts>0])),'sampled_unique_people':len(set(tr['patient'][counts>0])),
        'old_control_exact_state':exact_old,'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),'sampling_code_sha256':sha(ROOT/'scripts/skin_color_sampling.py'),
        'only_original_TRAIN_loaded':True,'reserved_endpoint_access':False,'source_exploratory_only':True,'risk_calibrated':False}
    write(out/'result.json',r);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':scores['full']['mean'],'p95':scores['full']['p95'],'at80':scores['coverage'][3]['mean'],'seconds':seconds}),flush=True)
    del model,opt,x,rgb,vx,vrgb,y;torch.cuda.empty_cache()
```

</details>
