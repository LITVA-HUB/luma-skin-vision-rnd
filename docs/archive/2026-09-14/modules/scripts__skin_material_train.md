# `scripts/skin_material_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_material_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen real-image material decoder versus matched tangent controls.

SHA-256 исходника: `b5d7a76a80e6db3a18a8d6207f82fa20eaa3fe7d4ed7634e4ea55fdade8dc4f5`. Строк: **130**.

## Зависимости

```python
import os
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_train_pixels import digest
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,rows,write,pair_error
from skin_capture_model import MODES
from skin_material_model import MaterialImage,ARMS
from skin_material_prior import PRIOR
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/skin_material_train.py#L19)

```python
OUT=ROOT/'docs/benchmarks/skin_material_image_v1'
```

[Строка 20](../../../../scripts/skin_material_train.py#L20)

```python
RUN=ROOT/'experiments/runs/skin_material_image_v1'
```

[Строка 21](../../../../scripts/skin_material_train.py#L21)

```python
PROTOCOL=ROOT/'docs/research/skin_material_image_protocol_v1.md'
```

[Строка 22](../../../../scripts/skin_material_train.py#L22)

```python
SEEDS=[17,29,43]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `prediction` | FunctionDef | См. реализацию | [L25](../../../../scripts/skin_material_train.py#L25) |
| `fit` | FunctionDef | См. реализацию | [L35](../../../../scripts/skin_material_train.py#L35) |
| `main` | FunctionDef | См. реализацию | [L105](../../../../scripts/skin_material_train.py#L105) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L35–102</summary>

```python
def fit(arm,seed,train,val,protocol,evaluation=None):
    name=f'{protocol}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        rec=json.loads((out/'result.json').read_bytes())
        assert rec['best_sha256']==sha(folder/'best.pt') and rec['final_sha256']==sha(folder/'final.pt')
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(train['patient']).isdisjoint(val['patient'])
    if evaluation is not None:assert set(train['device']).isdisjoint(evaluation['device']) and set(val['device']).isdisjoint(evaluation['device'])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=train['target'].mean(0).astype(np.float32);ys=train['target'].std(0).astype(np.float32)
    x=torch.from_numpy(train['tokens']).cuda();vx=torch.from_numpy(val['tokens']).cuda()
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    y=(torch.from_numpy(train['target']).cuda().float()-mean)/std
    mode=torch.tensor([MODES.index(str(v)) for v in train['mode']],device='cuda')
    model=MaterialImage(arm,dict(np.load(PRIOR)),ym,ys);initial=digest(model.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    steps=int(np.ceil(len(x)/32));history=[];best=float('inf');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'epoch':epoch,
            'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    for epoch in range(1,81):
        model.train();a,b=pair_indices(train['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[];norms=[]
        for offset in range(0,len(a),16):
            ix=torch.tensor(np.r_[a[offset:offset+16],b[offset:offset+16]],device='cuda')
            opt.zero_grad(set_to_none=True);p,g,h,r=model(x[ix])
            loss=(p-y[ix]).square().mean()+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            if arm.endswith('_residual'):loss=loss+.01*r.square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        scheduler.step();p,_,_,_=prediction(model,vx,mean,std);metric=summarize(delta_e00(p,val['target']),rows(val))
        value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p,target=val['target'])
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'max_gradient_norm':max(norms),'selection_mean':metric['mean'],'selection_patient_mean':value})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std)[0],target=val['target'])
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    assert np.array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=val if evaluation is None else evaluation;ex=vx if evaluation is None else torch.from_numpy(ev['tokens']).cuda()
    p,g,h,r=prediction(model,ex,mean,std);e=delta_e00(p,ev['target'])
    risk=np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1))
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],error=e,risk=risk,gate=g,hypotheses=h,
        residual_or_free_coordinates=r,patient=ev['patient'],site=ev['site'])
    order=np.argsort(risk,kind='stable')
    curve=np.cumsum(e[order])/np.arange(1,len(e)+1)
    np.savez(folder/'risk_curve.npz',coverage=np.arange(1,len(e)+1)/len(e),mean_error=curve)
    record={'arm':arm,'seed':seed,'protocol':protocol,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'full':summarize(e,rows(ev)),'paired_repeatability':pair_error(p,ev),
        'initial_state_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'stored_parameters':sum(q.numel() for q in model.parameters()),'active_parameters':model.active_parameters(),
        'model_bytes':(folder/'best.pt').stat().st_size,'peak_train_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'elapsed_seconds':time.perf_counter()-start,
        'mean_native_residual_or_free_rms':float(np.sqrt(np.mean(np.sum(r*r,axis=2),axis=1)).mean()),
        'train_people':len(set(train['patient'])),'selection_people':len(set(val['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(train['target']),'selection_images':len(val['target']),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(train['device']).tolist(),'selection_cameras':np.unique(val['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_script_sha256':sha(Path(__file__)),'model_script_sha256':sha(ROOT/'scripts/skin_material_model.py'),
        'source_exploratory_only':True,'reserved_test_or_calibration_loaded':False,
        'uncalibrated_disagreement_coverage':[]}
    for coverage in [1.,.95,.9,.8,.7,.6]:
        n=int(np.ceil(len(e)*coverage));ix=order[:n]
        record['uncalibrated_disagreement_coverage'].append({'requested_coverage':coverage,'accepted':n,**summarize(e[ix],rows(subset(ev,ix)))})
    record['strata']={k:{str(v):summarize(e[ev[k]==v],rows(subset(ev,ev[k]==v))) for v in np.unique(ev[k])} for k in ['device','image_type','mode']}
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':record['full']['mean'],'seconds':record['elapsed_seconds']}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()
```

</details>
