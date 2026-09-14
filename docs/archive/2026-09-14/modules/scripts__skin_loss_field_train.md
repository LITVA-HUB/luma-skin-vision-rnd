# `scripts/skin_loss_field_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_loss_field_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Train matched native-Lab point, categorical and candidate-loss fields.

SHA-256 исходника: `48c8e4e12494cfea9165a0871459f568399ec78d431319801238550b5985f5c8`. Строк: **160**.

## Зависимости

```python
import os
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from luma_skin_vision.color import delta_e00
from skin_loss_field import ARMS,LossFieldImage,risk_embedding,affine_weights
from skin_capture_model import MODES
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_train_pixels import digest
from skin_mskcc_summary_pilot import summarize
from skin_pair_train import subset,rows,write
from skin_pair_invariance import pair_indices
from skin_distribution_train import score
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 19](../../../../scripts/skin_loss_field_train.py#L19)

```python
OUT=ROOT/'docs/benchmarks/skin_loss_field_v1'
```

[Строка 20](../../../../scripts/skin_loss_field_train.py#L20)

```python
RUN=ROOT/'experiments/runs/skin_loss_field_v1'
```

[Строка 21](../../../../scripts/skin_loss_field_train.py#L21)

```python
PRIOR=ROOT/'artifacts/skin_loss_field_v1'
```

[Строка 22](../../../../scripts/skin_loss_field_train.py#L22)

```python
PROTOCOL=ROOT/'docs/research/skin_loss_field_protocol_v1.md'
```

[Строка 23](../../../../scripts/skin_loss_field_train.py#L23)

```python
SEEDS=(17,29,43)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `build_palette` | FunctionDef | См. реализацию | [L26](../../../../scripts/skin_loss_field_train.py#L26) |
| `prediction` | FunctionDef | См. реализацию | [L37](../../../../scripts/skin_loss_field_train.py#L37) |
| `gpu_palette` | FunctionDef | См. реализацию | [L51](../../../../scripts/skin_loss_field_train.py#L51) |
| `fit` | FunctionDef | См. реализацию | [L55](../../../../scripts/skin_loss_field_train.py#L55) |
| `main` | FunctionDef | См. реализацию | [L129](../../../../scripts/skin_loss_field_train.py#L129) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L55–126</summary>

```python
def fit(arm,seed,tr,va,protocol,evaluation=None):
    name=f'{protocol}__{arm}__s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes())
        assert sha(folder/'best.pt')==r['best_sha256'] and sha(folder/'final.pt')==r['final_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(tr['patient']).isdisjoint(va['patient'])
    if evaluation is not None:assert set(tr['device']).isdisjoint(evaluation['device'])
    palette=dict(np.load(PRIOR/f'{protocol}.npz'));atoms,index=np.unique(tr['target'],axis=0,return_inverse=True)
    np.testing.assert_array_equal(atoms,palette['atoms']);np.testing.assert_array_equal(index,palette['index'])
    torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    pal=gpu_palette(palette);ym=tr['target'].mean(0).astype(np.float32);ys=tr['target'].std(0).astype(np.float32)
    mean=torch.from_numpy(ym).cuda();std=torch.from_numpy(ys).cuda()
    x=torch.from_numpy(tr['tokens']).cuda();vx=torch.from_numpy(va['tokens']).cuda()
    y=(torch.from_numpy(tr['target']).cuda().float()-mean)/std
    ids=torch.from_numpy(index).cuda();target_phi=pal['phi'][ids];soft=pal['soft'][ids]
    mode=torch.tensor([MODES.index(str(m)) for m in tr['mode']],device='cuda')
    model=LossFieldImage(len(atoms));initial=digest(model.state_dict());model=model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80,eta_min=.00001)
    def state(epoch):
        return {'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'arm':arm,'seed':seed,'epoch':epoch,
            'target_mean':torch.from_numpy(ym),'target_std':torch.from_numpy(ys),
            'palette_sha256':sha(PRIOR/f'{protocol}.npz'),'protocol_sha256':sha(PROTOCOL)}
    best=float('inf');history=[];steps=int(np.ceil(len(x)/32));torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(1,81):
        model.train();a,b=pair_indices(tr['site'],steps*16,np.random.default_rng(seed*1000+epoch));losses=[]
        for offset in range(0,len(a),16):
            ix=torch.tensor(np.r_[a[offset:offset+16],b[offset:offset+16]],device='cuda')
            opt.zero_grad(set_to_none=True);point,g,h,logits=model(x[ix])
            if arm=='direct':loss=(point-y[ix]).square().mean()
            elif arm=='soft_ce':loss=-(soft[ix]*logits.log_softmax(1)).sum(1).mean()
            else:
                w=affine_weights(logits) if arm=='risk_affine' else logits.softmax(1)
                loss=(w@pal['phi']-target_phi[ix]).square().sum(1).mean()
            loss=loss+.1*torch.nn.functional.cross_entropy(g,mode[ix])
            if not torch.isfinite(loss):raise ValueError('Nonfinite field loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        scheduler.step();p=prediction(model,vx,mean,std,pal,arm)[0]
        metric=summarize(delta_e00(p,va['target']),rows(va));value=metric['patient_balanced_mean']
        if value<best:
            best=value;torch.save(state(epoch),folder/'best.pt');np.savez(folder/'best_selection.npz',prediction=p)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selection_patient_mean':value,'selection_mean':metric['mean']})
    torch.save(state(80),folder/'final.pt');np.savez(folder/'final_selection.npz',prediction=prediction(model,vx,mean,std,pal,arm)[0])
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**20
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['state'])
    np.testing.assert_array_equal(prediction(model,vx,mean,std,pal,arm)[0],np.load(folder/'best_selection.npz')['prediction'])
    ev=va if evaluation is None else evaluation
    p,secondary,risk,w,idx=prediction(model,torch.from_numpy(ev['tokens']).cuda(),mean,std,pal,arm)
    arrays={'prediction':p,'secondary':secondary,'risk':risk,'weights':w,'indices':idx,'target':ev['target'],'patient':ev['patient'],'site':ev['site']}
    scores={}
    for label,pred in [('primary',p),('secondary',secondary)]:
        scores[label],e,order=score(pred,risk,ev);arrays[label+'_error']=e
        arrays[label+'_curve']=np.cumsum(e[order])/np.arange(1,len(e)+1)
    np.savez(folder/'evaluation.npz',**arrays)
    record={'arm':arm,'seed':seed,'protocol':protocol,'best_epoch':saved['epoch'],'selection_patient_mean':best,
        'scores':scores,'initial_sha256':initial,'best_sha256':sha(folder/'best.pt'),'final_sha256':sha(folder/'final.pt'),
        'evaluation_sha256':sha(folder/'evaluation.npz'),'atoms':len(atoms),'candidates':len(palette['grid']),
        'stored_parameters':sum(p.numel() for p in model.parameters()),
        'active_parameters':sum(p.numel() for k,p in model.named_parameters() if not(arm=='direct' and k.startswith('atom_head.')))-(0 if arm=='direct' else 12*257),
        'checkpoint_bytes':(folder/'best.pt').stat().st_size,'fixed_palette_bytes':(PRIOR/f'{protocol}.npz').stat().st_size,
        'fit_peak_allocated_mib':peak,'fit_seconds':seconds,'negative_risk_fraction':float((risk<0).mean()),
        'negative_weight_fraction':float((w<0).mean()),'train_people':len(set(tr['patient'])),'selection_people':len(set(va['patient'])),
        'evaluation_people':len(set(ev['patient'])),'train_images':len(x),'selection_images':len(va['target']),'evaluation_images':len(ev['target']),
        'train_cameras':np.unique(tr['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),
        'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__)),'model_code_sha256':sha(ROOT/'scripts/skin_loss_field.py'),
        'palette_sha256':sha(PRIOR/f'{protocol}.npz'),'source_exploratory_only':True,'risk_calibrated':False,'reserved_endpoint_access':False}
    write(out/'result.json',record);write(out/'history.json',history)
    print(json.dumps({'completed':name,'primary':scores['primary']['full']['mean'],'secondary':scores['secondary']['full']['mean'],'seconds':seconds}),flush=True)
    del model,opt,x,vx,pal;torch.cuda.empty_cache()
```

</details>
