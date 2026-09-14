# `scripts/skin_distribution_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_distribution_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Matched real-image conditional native-Lab distribution experiment.

SHA-256 исходника: `9b3ef6b5a4f6c7813f33b528774a8461534d3c0d65bd74564b7bb1d53e7892e6`. Строк: **155**.

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
from skin_pair_train import subset,rows,write
from skin_capture_model import MODES
from skin_distribution_model import ARMS,ColorDistribution,density_nll,color_decision
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 18](../../../../scripts/skin_distribution_train.py#L18)

```python
OUT=ROOT/'docs/benchmarks/skin_distribution_v1'
```

[Строка 19](../../../../scripts/skin_distribution_train.py#L19)

```python
RUN=ROOT/'experiments/runs/skin_distribution_v1'
```

[Строка 20](../../../../scripts/skin_distribution_train.py#L20)

```python
PROTOCOL=ROOT/'docs/research/skin_distribution_protocol_v1.md'
```

[Строка 21](../../../../scripts/skin_distribution_train.py#L21)

```python
SEEDS=[17,29,43]
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `prediction` | FunctionDef | См. реализацию | [L24](../../../../scripts/skin_distribution_train.py#L24) |
| `endpoints` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_distribution_train.py#L34) |
| `score` | FunctionDef | См. реализацию | [L45](../../../../scripts/skin_distribution_train.py#L45) |
| `fit` | FunctionDef | См. реализацию | [L55](../../../../scripts/skin_distribution_train.py#L55) |
| `main` | FunctionDef | См. реализацию | [L133](../../../../scripts/skin_distribution_train.py#L133) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L55–130</summary>

```python
def fit(arm,seed,tr,va,protocol,evaluation=None):
    name=f'{protocol}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        rec=json.loads((out/'result.json').read_bytes())
        assert sha(folder/'best.pt')==rec['best_sha256'] and sha(folder/'final.pt')==rec['final_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(tr['patient']).isdisjoint(va['patient'])
    if evaluation is not None:
        assert set(tr['device']).isdisjoint(evaluation['device']) and set(va['device']).isdisjoint(evaluation['device'])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();vx=torch.from_numpy(va['tokens']).cuda()
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    modes=torch.tensor([MODES.index(str(v)) for v in tr['mode']],device='cuda')
    model=ColorDistribution();initial=digest(model.state_dict())
    base_initial=digest({k:v for k,v in model.state_dict().items() if not k.startswith('scale_head.')})
    model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'arm':arm,'seed':seed,'epoch':epoch,'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),
            'protocol_sha256':sha(PROTOCOL)}
    steps=int(np.ceil(len(x)/32));best=float('inf');history=[]
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(1,81):
        model.train();a,b=pair_indices(tr['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[]
        for offset in range(0,len(a),16):
            ix=torch.tensor(np.r_[a[offset:offset+16],b[offset:offset+16]],device='cuda')
            opt.zero_grad(set_to_none=True);p,g,h,s=model(x[ix])
            if arm in ('mse','mse_mode'):
                loss=(p-y[ix]).square().mean()
                if arm=='mse_mode':loss=loss+.1*torch.nn.functional.cross_entropy(g,modes[ix])
            elif arm=='gaussian':loss=(2/3)*density_nll(y[ix],p[:,None],s.mean(1,keepdim=True),g[:,:1]*0).mean()
            else:loss=(2/3)*density_nll(y[ix],h,s,g).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite density objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=prediction(model,vx,mean,std)[0]
        metric=summarize(delta_e00(p,va['target']),rows(va));value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_patient_mean':value,'selection_mean':metric['mean']})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std)[0])
    elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    np.testing.assert_array_equal(prediction(model,vx,mean,std)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=va if evaluation is None else evaluation;ex=vx if evaluation is None else torch.from_numpy(ev['tokens']).cuda()
    p,g,h,s=prediction(model,ex,mean,std);ep,decision=endpoints(arm,p,g,h,s)
    arrays={'point':p,'gate':g,'hypotheses':h,'scales':s,'target':ev['target'],'patient':ev['patient'],'site':ev['site']}
    scores={}
    for label,(pred,risk) in ep.items():
        scores[label],e,order=score(pred,risk,ev)
        arrays.update({label+'_prediction':pred,label+'_risk':risk,label+'_error':e,
            label+'_curve':np.cumsum(e[order])/np.arange(1,len(e)+1)})
    np.savez(folder/'evaluation.npz',**arrays)
    record={'arm':arm,'seed':seed,'protocol':protocol,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'scores':scores,'initial_sha256':initial,'base_initial_sha256':base_initial,
        'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'evaluation_sha256':sha(folder/'evaluation.npz'),'stored_parameters':sum(p.numel() for p in model.parameters()),
        'active_parameters':sum(p.numel() for k,p in model.named_parameters() if not(arm in ('mse','mse_mode') and k.startswith('scale_head.'))),
        'checkpoint_bytes':(folder/'best.pt').stat().st_size,'fit_peak_allocated_mib':peak,'fit_seconds':elapsed,
        'train_people':len(set(tr['patient'])),'selection_people':len(set(va['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(x),'selection_images':len(vx),'evaluation_images':len(ex),
        'train_cameras':np.unique(tr['device']).tolist(),'selection_cameras':np.unique(va['device']).tolist(),
        'evaluation_cameras':np.unique(ev['device']).tolist(),'source_exploratory_only':True,
        'risk_calibrated':False,'reserved_endpoint_access':False,
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),
        'model_code_sha256':sha(ROOT/'scripts/skin_distribution_model.py')}
    if decision:
        record['quadrature_decision2_vs3_mean_delta_e00']=float(delta_e00(ep['decision2'][0],ep['decision3'][0]).mean())
        record['fraction_decision3_changes_mean']=float((delta_e00(ep['mean'][0],ep['decision3'][0])>.01).mean())
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'means':{k:v['full']['mean'] for k,v in scores.items()},'seconds':elapsed}),flush=True)
    del model,opt,x,vx,y;torch.cuda.empty_cache()
```

</details>
