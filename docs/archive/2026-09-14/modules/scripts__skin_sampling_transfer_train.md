# `scripts/skin_sampling_transfer_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_sampling_transfer_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Fixed-budget source camera-transfer allocation experiment; final checkpoint only.

SHA-256 исходника: `f72f2fac96331297f41569c2c89c9cb2abd77fc101f48916e6dac8033bd7744d`. Строк: **99**.

## Зависимости

```python
import os
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_sampling_transfer import ARMS,sampling_distribution,draw_indices,protocols
from skin_support_curve import SkinRepresentation
from skin_support_curve_train import novelty
from skin_color_sampling_train import bindings as base_bindings
from skin_capture_model import MODES
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_train_pixels import digest
from skin_pair_train import write
from skin_distribution_train import score
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/skin_sampling_transfer_train.py#L19)

```python
OUT=ROOT/'docs/benchmarks/skin_sampling_transfer_v1'
```

[Строка 19](../../../../scripts/skin_sampling_transfer_train.py#L19)

```python
RUN=ROOT/'experiments/runs/skin_sampling_transfer_v1'
```

[Строка 20](../../../../scripts/skin_sampling_transfer_train.py#L20)

```python
PROTOCOL=ROOT/'docs/research/skin_sampling_transfer_protocol_v1.md'
```

[Строка 20](../../../../scripts/skin_sampling_transfer_train.py#L20)

```python
SEEDS=(17,29,43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `predict` | FunctionDef | См. реализацию | [L23](../../../../scripts/skin_sampling_transfer_train.py#L23) |
| `fit` | FunctionDef | См. реализацию | [L28](../../../../scripts/skin_sampling_transfer_train.py#L28) |
| `bindings` | FunctionDef | См. реализацию | [L80](../../../../scripts/skin_sampling_transfer_train.py#L80) |
| `main` | FunctionDef | См. реализацию | [L85](../../../../scripts/skin_sampling_transfer_train.py#L85) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L23–25</summary>

```python
def predict(model,tokens,mean,std):
    model.eval()
    with torch.no_grad():return np.concatenate([(model(t,None)[0]*std+mean).cpu().numpy() for t in tokens.split(32)])
```

</details>

<details><summary>fit · L28–77</summary>

```python
def fit(protocol,tr,evaluations,arm,seed):
    name=f'{protocol}__{arm}__s{seed}';out=OUT/name;folder=RUN/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes());assert sha(folder/'final.pt')==r['checkpoint_sha256']
        for domain,v in r['evaluations'].items():assert sha(folder/(domain+'.npz'))==v['array_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    out.mkdir(parents=True,exist_ok=False);folder.mkdir(parents=True,exist_ok=False)
    for domain,ev in evaluations.items():
        assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
        if domain=='unseen':assert set(tr['device']).isdisjoint(ev['device'])
    q,correction,info=sampling_distribution(tr,arm)
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=SkinRepresentation('baseline');initial=digest(model.state_dict());model=model.cuda()
    ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda();x=torch.from_numpy(tr['tokens']).cuda()
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda');importance=torch.tensor(correction,dtype=torch.float32,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
    trace=hashlib.sha256();counts=np.zeros(len(q),np.int64);history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,31):
        model.train();indices=draw_indices(q,arm,seed,epoch);trace.update(indices.tobytes());counts+=np.bincount(indices.ravel(),minlength=len(q));losses=[]
        for batch in indices:
            ix=torch.from_numpy(batch).cuda();opt.zero_grad(set_to_none=True);p,g,_=model(x[ix],None)
            if arm=='color_ipw':loss=(((p-y[ix]).square().mean(1)+.1*torch.nn.functional.cross_entropy(g,mode[ix],reduction='none'))*importance[ix]).mean()
            else:loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        scheduler.step();history.append({'round':epoch,'loss':float(np.mean(losses))})
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    state={'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'protocol':protocol,
        'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    torch.save(state,folder/'final.pt');np.savez(folder/'sampling.npz',proposal=q,correction=correction,counts=counts,site_probability=info['site_probability'],density=info['density'])
    scored={}
    for domain,ev in evaluations.items():
        p=predict(model,torch.from_numpy(ev['tokens']).cuda(),mean,std);risk=novelty(tr,ev);scores,error,order=score(p,risk,ev)
        scores['site_balanced_mean']=float(np.mean([error[ev['site']==s].mean() for s in np.unique(ev['site'])]))
        np.savez(folder/(domain+'.npz'),prediction=p,target=ev['target'],patient=ev['patient'],site=ev['site'],risk=risk,error=error,curve=np.cumsum(error[order])/np.arange(1,len(error)+1))
        scored[domain]={'scores':scores,'people':len(set(ev['patient'])),'images':len(error),'cameras':np.unique(ev['device']).tolist(),'array_sha256':sha(folder/(domain+'.npz'))}
    r={'protocol':protocol,'arm':arm,'seed':seed,'train_people':len(set(tr['patient'])),'train_images':len(x),'train_sites':info['site_count'],'train_cameras':np.unique(tr['device']).tolist(),
        'evaluations':scored,'rounds':30,'optimizer_updates':930,'sampled_images':int(counts.sum()),'draw_sha256':trace.hexdigest(),
        'initial_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),'fit_seconds':seconds,'fit_peak_allocated_mib':peak,'checkpoint_bytes':(folder/'final.pt').stat().st_size,
        'checkpoint_sha256':sha(folder/'final.pt'),'sampling_sha256':sha(folder/'sampling.npz'),'effective_image_mass':float(1/np.sum(q*q)),
        'proposal_vs_uniform_site_min':float(info['site_probability'].min()*info['site_count']),'proposal_vs_uniform_site_max':float(info['site_probability'].max()*info['site_count']),
        'importance_min':float(correction.min()),'importance_max':float(correction.max()),'sampled_unique_images':int(np.sum(counts>0)),
        'sampled_unique_sites':len(set(tr['site'][counts>0])),'sampled_unique_people':len(set(tr['patient'][counts>0])),
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),'sampling_code_sha256':sha(ROOT/'scripts/skin_sampling_transfer.py'),
        'source_exploratory_only':True,'reserved_endpoint_access':False,'validation_used_for_fitting_or_selection':False,'risk_calibrated':False}
    write(out/'result.json',r);write(out/'history.json',history)
    print(json.dumps({'completed':name,'means':{k:v['scores']['full']['mean'] for k,v in scored.items()},'seconds':seconds}),flush=True)
    del model,opt,x,y;torch.cuda.empty_cache()
```

</details>
