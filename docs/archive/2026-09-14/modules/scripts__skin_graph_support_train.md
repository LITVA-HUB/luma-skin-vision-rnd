# `scripts/skin_graph_support_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_graph_support_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Spatially valid two-pass graph and observed-support experiment.

SHA-256 исходника: `90ae6f89a9d46d043409df74fd487f3f9693bfa49af2ba929ea71626763c15e9`. Строк: **130**.

## Зависимости

```python
import os
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from skin_capture_model import CaptureColor,MODES
from skin_capture_support import make_plan,apply_plan
from skin_graph_support_model import GraphSupportColor,ARMS
from skin_expert_anchor_train import common_risk
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,rows,write
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_summary_pilot import summarize
from skin_distribution_train import score
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 21](../../../../scripts/skin_graph_support_train.py#L21)

```python
OUT=ROOT/'docs/benchmarks/skin_graph_support_v1'
```

[Строка 22](../../../../scripts/skin_graph_support_train.py#L22)

```python
RUN=ROOT/'experiments/runs/skin_graph_support_v1'
```

[Строка 23](../../../../scripts/skin_graph_support_train.py#L23)

```python
PROTOCOL=ROOT/'docs/research/skin_graph_support_protocol_v1.md'
```

[Строка 24](../../../../scripts/skin_graph_support_train.py#L24)

```python
SEEDS=(17,29,43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `update_digest` | FunctionDef | См. реализацию | [L27](../../../../scripts/skin_graph_support_train.py#L27) |
| `prediction` | FunctionDef | См. реализацию | [L31](../../../../scripts/skin_graph_support_train.py#L31) |
| `fit` | FunctionDef | См. реализацию | [L38](../../../../scripts/skin_graph_support_train.py#L38) |
| `main` | FunctionDef | См. реализацию | [L109](../../../../scripts/skin_graph_support_train.py#L109) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L38–106</summary>

```python
def fit(arm,seed,tr,va,protocol,evaluation=None):
    branch,augmentation=arm.split('_');support_arm='baseline' if augmentation=='raw' else 'paired_stratified'
    name=f'{protocol}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes());assert sha(folder/'best.pt')==r['best_sha256'] and sha(folder/'final.pt')==r['final_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(tr['patient']).isdisjoint(va['patient'])
    if evaluation is not None:assert set(tr['device']).isdisjoint(evaluation['device'])
    for site in np.unique(tr['site']):assert np.all(tr['target'][tr['site']==site]==tr['target'][tr['site']==site][0])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();vx=torch.from_numpy(va['tokens']).cuda()
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    soft_mode=torch.nn.functional.one_hot(mode,len(MODES)).float();partner=torch.tensor(np.r_[16:32,0:16],device='cuda')
    model=GraphSupportColor();initial=digest(model.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'epoch':epoch,
            'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    steps=int(np.ceil(len(x)/32));history=[];best=float('inf');trace=hashlib.sha256();pair_trace=hashlib.sha256()
    counts={'processed_bags':0,'partner_patch_slots':0,'softened_mode_bags':0,'requested_input_augmented_bags':0}
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(1,81):
        model.train();a,b=pair_indices(tr['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[]
        pair_trace.update(a.tobytes());pair_trace.update(b.tobytes());rng=np.random.default_rng(seed*100000+epoch)
        for offset in range(0,len(a),16):
            indices=np.r_[a[offset:offset+16],b[offset:offset+16]]
            assert np.array_equal(tr['site'][indices[:16]],tr['site'][indices[16:]])
            ix=torch.tensor(indices,device='cuda');plan=make_plan(32,64,rng);update_digest(trace,plan)
            inp,target,origin,index=apply_plan(x[ix],soft_mode[ix],partner,plan,support_arm)
            counts['processed_bags']+=32;counts['partner_patch_slots']+=int(origin.sum())
            counts['softened_mode_bags']+=int((target!=soft_mode[ix]).any(1).sum())
            if augmentation=='paired':counts['requested_input_augmented_bags']+=int(plan['augment'].sum())
            opt.zero_grad(set_to_none=True)
            first=model(x[ix],branch=branch=='graph')[0]
            second=model(inp,branch=False)[0]
            loss=.5*(first-y[ix]).square().mean()+.5*(second-y[ix]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite support loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=prediction(model,vx,mean,std)
        metric=summarize(delta_e00(p,va['target']),rows(va));value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_patient_mean':value,'selection_mean':metric['mean']})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std))
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    np.testing.assert_array_equal(prediction(model,vx,mean,std),np.load(folder/'best_selection.npz')['prediction'])
    ev=va if evaluation is None else evaluation
    p=prediction(model,torch.from_numpy(ev['tokens']).cuda(),mean,std)
    risk=common_risk(tr,ev);scores,e,order=score(p,risk,ev);scores.pop('predicted_error_mae_or_dispersion_mae')
    np.savez(folder/'evaluation.npz',prediction=p,risk=risk,error=e,curve=np.cumsum(e[order])/np.arange(1,len(e)+1),
        target=ev['target'],patient=ev['patient'],site=ev['site'])
    r={'arm':arm,'seed':seed,'protocol':protocol,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'scores':scores,'initial_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'evaluation_sha256':sha(folder/'evaluation.npz'),'plan_sha256':trace.hexdigest(),'pair_draw_sha256':pair_trace.hexdigest(),
        'augmentation_counts':counts,'parameters':sum(q.numel() for q in model.parameters()),'inference_parameters':model.inference_parameters(),'training_active_parameters':model.active_parameters() if branch=='graph' else model.inference_parameters(),'graph_at_inference':False,'passes_per_step':2,'checkpoint_bytes':(folder/'best.pt').stat().st_size,
        'fit_peak_allocated_mib':peak,'fit_seconds':seconds,'train_people':len(set(tr['patient'])),'selection_people':len(set(va['patient'])),
        'evaluation_people':len(set(ev['patient'])),'train_images':len(x),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(tr['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),'augmentation_code_sha256':sha(ROOT/'scripts/skin_capture_support.py'),
        'source_exploratory_only':True,'risk_calibrated':False,'reserved_endpoint_access':False}
    write(out/'result.json',r);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':scores['full']['mean'],'at80':scores['coverage'][3]['mean'],'seconds':seconds}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()
```

</details>
