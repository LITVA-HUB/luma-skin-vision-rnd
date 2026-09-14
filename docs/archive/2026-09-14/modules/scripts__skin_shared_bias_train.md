# `scripts/skin_shared_bias_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_shared_bias_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Predeclared paired common-bias objectives; unchanged single-image architecture.

SHA-256 исходника: `699bb88d725b462d5d321fb1d02bdc47bc87a3ff293f176b8b15dfa52e58f9ba`. Строк: **119**.

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
from skin_capture_model import CaptureColor,MODES
from skin_shared_bias import ARMS,paired_objective
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/skin_shared_bias_train.py#L18)

```python
OUT=ROOT/'docs/benchmarks/skin_shared_bias_v1'
```

[Строка 19](../../../../scripts/skin_shared_bias_train.py#L19)

```python
RUN=ROOT/'experiments/runs/skin_shared_bias_v1'
```

[Строка 20](../../../../scripts/skin_shared_bias_train.py#L20)

```python
PROTOCOL=ROOT/'docs/research/skin_shared_bias_protocol_v1.md'
```

[Строка 21](../../../../scripts/skin_shared_bias_train.py#L21)

```python
SEEDS=[17,29,43]
```

[Строка 22](../../../../scripts/skin_shared_bias_train.py#L22)

```python
OBJECTIVES=ARMS
```

[Строка 23](../../../../scripts/skin_shared_bias_train.py#L23)

```python
ARCHES=['mixture']
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `prediction` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_shared_bias_train.py#L26) |
| `fit` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_shared_bias_train.py#L34) |
| `main` | FunctionDef | См. реализацию | [L98](../../../../scripts/skin_shared_bias_train.py#L98) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L34–95</summary>

```python
def fit(arch,objective,seed,train,val,protocol_name,evaluation=None):
    assert set(train['patient']).isdisjoint(val['patient'])
    if evaluation is not None:
        assert set(train['device']).isdisjoint(evaluation['device']) and set(val['device']).isdisjoint(evaluation['device'])
    name=f'{protocol_name}__{arch}_{objective}__s{seed}';folder=RUN/name;out=OUT/name
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=train['target'].mean(0).astype(np.float32);ys=train['target'].std(0).astype(np.float32)
    x=torch.from_numpy(train['tokens']).cuda();vx=torch.from_numpy(val['tokens']).cuda()
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda();native=torch.from_numpy(train['target']).cuda()
    y=(native.float()-mean)/std;mode=torch.tensor([MODES.index(str(v)) for v in train['mode']],device='cuda')
    model=CaptureColor(arch);initial=digest(model.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    steps=int(np.ceil(len(x)/32));history=[];best=float('inf');start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arch':arch,'objective':objective,'seed':seed,'epoch':epoch,
                'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),'protocol_sha256':sha(PROTOCOL)}
    for epoch in range(1,81):
        model.train();a,b=pair_indices(train['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[];norms=[]
        for offset in range(0,len(a),16):
            ix=torch.tensor(np.r_[a[offset:offset+16],b[offset:offset+16]],device='cuda')
            opt.zero_grad(set_to_none=True);pred,logits,_=model(x[ix])
            color=paired_objective(pred,y[ix],objective)
            loss=color if arch=='plain' else color+.1*torch.nn.functional.cross_entropy(logits,mode[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        scheduler.step();p,_,_=prediction(model,vx,mean,std);metric=summarize(delta_e00(p,val['target']),rows(val))
        value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p,target=val['target'])
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'max_gradient_norm':max(norms),'selection_mean':metric['mean'],'selection_patient_mean':value})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std)[0],target=val['target'])
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    assert np.array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=val if evaluation is None else evaluation;ex=vx if evaluation is None else torch.from_numpy(ev['tokens']).cuda()
    p,g,h=prediction(model,ex,mean,std);e=delta_e00(p,ev['target']);m=np.array([MODES.index(str(v)) for v in ev['mode']])
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],error=e,gate=g,hypotheses=h,patient=ev['patient'],site=ev['site'])
    record={'arch':arch,'objective':objective,'seed':seed,'protocol':protocol_name,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'full':summarize(e,rows(ev)),'paired_repeatability':pair_error(p,ev),'mode_accuracy':float((g.argmax(1)==m).mean()),
        'mean_gate_entropy':float(-(g*np.log(np.maximum(g,1e-12))).sum(1).mean()),
        'mean_hypothesis_rms_delta_e76':float(np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1)).mean()),
        'initial_state_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'stored_parameters':sum(p.numel() for p in model.parameters()),'gate_parameters':sum(p.numel() for p in model.gate.parameters()),
        'peak_train_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':time.perf_counter()-start,
        'train_people':len(set(train['patient'])),'selection_people':len(set(val['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(train['target']),'selection_images':len(val['target']),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(train['device']).tolist(),'selection_cameras':np.unique(val['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_script_sha256':sha(Path(__file__)),'model_script_sha256':sha(ROOT/'scripts/skin_capture_model.py'),
        'source_exploratory_only':True,'reserved_test_or_calibration_loaded':False}
    risk=np.sqrt(np.mean(np.sum((h-h.mean(1,keepdims=True))**2,axis=2),axis=1))
    record['uncalibrated_disagreement_coverage']=[]
    order=np.argsort(risk,kind='stable')
    for coverage in [1.,.95,.9,.8,.7,.6]:
        n=int(np.ceil(len(e)*coverage));ix=order[:n]
        record['uncalibrated_disagreement_coverage'].append({'requested_coverage':coverage,'accepted':n,**summarize(e[ix],rows(subset(ev,ix)))})
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],error=e,risk=risk,gate=g,hypotheses=h,patient=ev['patient'],site=ev['site'])
    record['strata']={k:{str(v):summarize(e[ev[k]==v],rows(subset(ev,ev[k]==v))) for v in np.unique(ev[k])} for k in ['device','image_type','mode']}
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':record['full']['mean'],'mode_accuracy':record['mode_accuracy'],'seconds':record['elapsed_seconds']}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()
```

</details>
