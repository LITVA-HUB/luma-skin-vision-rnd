# `scripts/skin_copula_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_copula_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Frozen real-photo source screen; no reserved endpoints or spectral pseudo-labels.

SHA-256 исходника: `0fded8f56438d01f0d0a65545ddd0cbe1491ee99097e8222e9fd79827a42d620`. Строк: **125**.

## Зависимости

```python
import os
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_copula_data import load,pack
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_train_pixels import digest
from skin_pair_invariance import pair_indices
from skin_pair_train import subset,rows,write,pair_error
from skin_copula_model import DistributionColor as SpatialColor,ARMS
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 17](../../../../scripts/skin_copula_train.py#L17)

```python
OUT=ROOT/'docs/benchmarks/skin_copula_v1'
```

[Строка 18](../../../../scripts/skin_copula_train.py#L18)

```python
RUN=ROOT/'experiments/runs/skin_copula_v1'
```

[Строка 19](../../../../scripts/skin_copula_train.py#L19)

```python
PROTOCOL=ROOT/'docs/research/skin_copula_protocol_v1.md'
```

[Строка 20](../../../../scripts/skin_copula_train.py#L20)

```python
SEEDS=[17,29,43]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `prediction` | FunctionDef | См. реализацию | [L23](../../../../scripts/skin_copula_train.py#L23) |
| `selective` | FunctionDef | См. реализацию | [L33](../../../../scripts/skin_copula_train.py#L33) |
| `fit` | FunctionDef | См. реализацию | [L42](../../../../scripts/skin_copula_train.py#L42) |
| `main` | FunctionDef | См. реализацию | [L101](../../../../scripts/skin_copula_train.py#L101) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L42–98</summary>

```python
def fit(arm,seed,train,val,protocol_name,evaluation=None):
    assert set(train['patient']).isdisjoint(val['patient'])
    if evaluation is not None:
        assert set(train['patient']).isdisjoint(evaluation['patient'])
        assert set(train['device']).isdisjoint(evaluation['device']) and set(val['device']).isdisjoint(evaluation['device'])
    name=f'{protocol_name}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=train['target'].mean(0).astype(np.float32);ys=train['target'].std(0).astype(np.float32)
    assert np.all(ys>0)
    x=torch.from_numpy(pack(train,arm)).cuda();vx=torch.from_numpy(pack(val,arm)).cuda()
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    y=(torch.from_numpy(train['target']).float().cuda()-mean)/std
    model=SpatialColor(arm);initial=digest(model.state_dict());model=model.cuda()
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
            opt.zero_grad(set_to_none=True);pred,_,_=model(x[ix]);loss=(pred-y[ix]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()));norms.append(float(norm))
        scheduler.step();p,risk=prediction(model,vx,mean,std);metric=summarize(delta_e00(p,val['target']),rows(val))
        value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt')
            np.savez(folder/'best_selection.npz',prediction=p,risk=risk,target=val['target'])
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'max_gradient_norm':max(norms),'selection_mean':metric['mean'],'selection_patient_mean':value})
    torch.save(state(80),folder/'final.pt');p,risk=prediction(model,vx,mean,std)
    np.savez(folder/'final_selection.npz',prediction=p,risk=risk,target=val['target'])
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    assert np.array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    train_peak=torch.cuda.max_memory_allocated()/2**20
    ev=val if evaluation is None else evaluation;ex=vx if evaluation is None else torch.from_numpy(pack(ev,arm)).cuda()
    p,risk=prediction(model,ex,mean,std);e=delta_e00(p,ev['target'])
    np.savez(folder/'evaluation.npz',prediction=p,target=ev['target'],error=e,risk=risk,patient=ev['patient'],site=ev['site'])
    record={'arm':arm,'seed':seed,'protocol':protocol_name,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'full':summarize(e,rows(ev)),'uncalibrated_dispersion_coverage':selective(e,risk,ev),'paired_repeatability':pair_error(p,ev),
        'initial_state_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'stored_parameters':sum(p.numel() for p in model.parameters()),'active_parameters':model.active_parameters(),
        'peak_train_allocated_mib':train_peak,'elapsed_seconds':time.perf_counter()-start,
        'train_people':len(set(train['patient'])),'selection_people':len(set(val['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(train['target']),'selection_images':len(val['target']),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(train['device']).tolist(),'selection_cameras':np.unique(val['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_script_sha256':sha(Path(__file__)),'model_script_sha256':sha(ROOT/'scripts/skin_copula_model.py'),
        'source_exploratory_only':True,'reserved_test_or_calibration_loaded':False,
        'risk_is_calibrated_expected_error':False}
    record['strata']={k:{str(v):summarize(e[ev[k]==v],rows(subset(ev,ev[k]==v))) for v in np.unique(ev[k])} for k in ['device','image_type','mode']}
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'mean':record['full']['mean'],'risk80':record['uncalibrated_dispersion_coverage'][3]['mean'],'seconds':record['elapsed_seconds']}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()
```

</details>
