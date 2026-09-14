# `scripts/skin_support_curve_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_support_curve_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Equal-update TRAIN-only skin representation learning curves.

SHA-256 исходника: `09b3dbf91c975dcb7f9d98631497c41f5ec1902977a37cc31fe71a9752080339`. Строк: **118**.

## Зависимости

```python
import os
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_support_curve import ARMS,SkinRepresentation,patient_roles
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

[Строка 18](../../../../scripts/skin_support_curve_train.py#L18)

```python
OUT=ROOT/'docs/benchmarks/skin_support_curve_v1'
```

[Строка 19](../../../../scripts/skin_support_curve_train.py#L19)

```python
RUN=ROOT/'experiments/runs/skin_support_curve_v1'
```

[Строка 20](../../../../scripts/skin_support_curve_train.py#L20)

```python
PROTOCOL=ROOT/'docs/research/skin_support_curve_protocol_v1.md'
```

[Строка 21](../../../../scripts/skin_support_curve_train.py#L21)

```python
SEEDS=(17,29,43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `novelty` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_support_curve_train.py#L24) |
| `predict` | FunctionDef | См. реализацию | [L30](../../../../scripts/skin_support_curve_train.py#L30) |
| `role_digest` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_support_curve_train.py#L38) |
| `fit` | FunctionDef | См. реализацию | [L42](../../../../scripts/skin_support_curve_train.py#L42) |
| `bindings` | FunctionDef | См. реализацию | [L96](../../../../scripts/skin_support_curve_train.py#L96) |
| `main` | FunctionDef | См. реализацию | [L104](../../../../scripts/skin_support_curve_train.py#L104) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L30–35</summary>

```python
def predict(model,tokens,rgb,mean,std):
    model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(tokens),32):
            values.append((model(tokens[start:start+32],rgb[start:start+32])[0]*std+mean).cpu().numpy())
    return np.concatenate(values)
```

</details>

<details><summary>fit · L42–93</summary>

```python
def fit(data,arm,count,seed):
    name=f'n{count}__{arm}__s{seed}';out=OUT/name;folder=RUN/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes())
        assert sha(folder/'final.pt')==r['checkpoint_sha256'] and sha(folder/'evaluation.npz')==r['evaluation_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    out.mkdir(parents=True,exist_ok=False);folder.mkdir(parents=True,exist_ok=False)
    a,b=patient_roles(data['patient'],data['device'],count,seed);tr,ev=subset(data,a),subset(data,b)
    assert set(tr['patient']).isdisjoint(ev['patient']) and set(tr['site']).isdisjoint(ev['site'])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=SkinRepresentation(arm);initial=digest(model.state_dict());core_initial=digest(model.core.state_dict());model=model.cuda()
    ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();rgb=torch.from_numpy(tr['rgb']).cuda().float()/255
    vx=torch.from_numpy(ev['tokens']).cuda();vrgb=torch.from_numpy(ev['rgb']).cuda().float()/255
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,30,eta_min=.00001)
    draw_hash=hashlib.sha256();history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,31):
        model.train();indices=np.random.default_rng(seed*1000+epoch).integers(0,len(x),size=(31,32),dtype=np.int64)
        draw_hash.update(indices.tobytes());losses=[]
        for batch in indices:
            ix=torch.from_numpy(batch).cuda();opt.zero_grad(set_to_none=True)
            p,g,_=model(x[ix],rgb[ix]);loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=predict(model,vx,vrgb,mean,std);metrics=summarize(delta_e00(p,ev['target']),rows(ev))
        history.append({'round':epoch,'loss':float(np.mean(losses)),'holdout_mean':metrics['mean'],'holdout_patient_mean':metrics['patient_balanced_mean']})
    peak=torch.cuda.max_memory_allocated()/2**20;seconds=time.perf_counter()-start
    checkpoint={'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'count':count,'seed':seed,
        'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    torch.save(checkpoint,folder/'final.pt');risk=novelty(tr,ev);scores,error,order=score(p,risk,ev)
    site_idx=np.unique(tr['site'],return_index=True)[1];palette=tr['target'][site_idx]
    support=delta_e00(ev['target'][:,None],palette[None]).min(1)
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],patient=ev['patient'],site=ev['site'],risk=risk,error=error,
        curve=np.cumsum(error[order])/np.arange(1,len(error)+1),reference_support_diagnostic=support)
    r={'arm':arm,'seed':seed,'train_people':count,'holdout_people':len(set(ev['patient'])),'train_images':len(x),'holdout_images':len(vx),
        'train_sites':len(palette),'holdout_sites':len(set(ev['site'])),'camera_people':{c:len(set(tr['patient'][tr['device']==c])) for c in ('SLR','ipod')},
        'train_target_min':tr['target'].min(0).tolist(),'train_target_max':tr['target'].max(0).tolist(),
        'support_diagnostic_mean':float(support.mean()),'support_diagnostic_p95':float(np.quantile(support,.95)),
        'scores':scores,'rounds':30,'optimizer_updates':930,'sampled_images':29760,'draw_sha256':draw_hash.hexdigest(),'role_sha256':role_digest(a,b),
        'initial_sha256':initial,'core_initial_sha256':core_initial,'parameters':sum(p.numel() for p in model.parameters()),
        'fit_seconds':seconds,'fit_peak_allocated_mib':peak,'checkpoint_bytes':(folder/'final.pt').stat().st_size,
        'checkpoint_sha256':sha(folder/'final.pt'),'evaluation_sha256':sha(folder/'evaluation.npz'),
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),'model_code_sha256':sha(ROOT/'scripts/skin_support_curve.py'),
        'source_exploratory_only':True,'only_original_TRAIN_loaded':True,'risk_calibrated':False,'reserved_endpoint_access':False}
    write(out/'result.json',r);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':scores['full']['mean'],'at80':scores['coverage'][3]['mean'],'seconds':seconds}),flush=True)
    del model,opt,x,rgb,vx,vrgb,y;torch.cuda.empty_cache()
```

</details>
