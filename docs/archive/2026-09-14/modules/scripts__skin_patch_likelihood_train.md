# `scripts/skin_patch_likelihood_train.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_patch_likelihood_train.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Native skin color from conditional patch distributions; source-only screen.

SHA-256 исходника: `0774327f89c8cd153f497b882c3700267e1c2126557855fb053fd288bd852faa`. Строк: **82**.

## Зависимости

```python
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_patch_likelihood import fit_patch,likelihood,color_output
from skin_appearance_inverse_train import novelty
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_pair_train import subset,rows,write
from skin_mskcc_summary_pilot import summarize
from skin_distribution_train import score
from luma_skin_vision.color import delta_e00
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 15](../../../../scripts/skin_patch_likelihood_train.py#L15)

```python
OUT=ROOT/'docs/benchmarks/skin_patch_likelihood_v1'
```

[Строка 16](../../../../scripts/skin_patch_likelihood_train.py#L16)

```python
RUN=ROOT/'experiments/runs/skin_patch_likelihood_v1'
```

[Строка 17](../../../../scripts/skin_patch_likelihood_train.py#L17)

```python
PROTOCOL=ROOT/'docs/research/skin_patch_likelihood_protocol_v1.md'
```

[Строка 18](../../../../scripts/skin_patch_likelihood_train.py#L18)

```python
STRENGTHS=(1,4,16,64)
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `configs` | FunctionDef | См. реализацию | [L21](../../../../scripts/skin_patch_likelihood_train.py#L21) |
| `fit` | FunctionDef | См. реализацию | [L28](../../../../scripts/skin_patch_likelihood_train.py#L28) |
| `main` | FunctionDef | См. реализацию | [L63](../../../../scripts/skin_patch_likelihood_train.py#L63) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>fit · L28–60</summary>

```python
def fit(protocol,t,v,ev,representation,degree,components,seed):
    name=f'{protocol}__{representation}_d{degree}_k{components}_s{seed}';folder=RUN/name;out=OUT/name
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_bytes());assert sha(folder/'model.npz')==r['model_sha256'];assert sha(folder/'evaluation.npz')==r['evaluation_sha256']
        print(json.dumps({'verified_existing':name}),flush=True);return
    folder.mkdir(parents=True,exist_ok=False);out.mkdir(parents=True,exist_ok=False)
    assert set(t['patient']).isdisjoint(v['patient']) and set(t['patient']).isdisjoint(ev['patient'])
    if protocol!='mixed':assert set(t['device']).isdisjoint(ev['device']) and set(v['device']).isdisjoint(ev['device'])
    start=time.perf_counter();model,history=fit_patch(t,representation,degree,components,seed);fit_seconds=time.perf_counter()-start
    np.savez(folder/'model.npz',**model);write(out/'history.json',history)
    vl=likelihood(model,v['tokens']);selection={};vp={}
    for strength in STRENGTHS:
        p,_=color_output(vl,model['palette'],strength);vp[f't{strength}']=p
        selection[str(strength)]=summarize(delta_e00(p,v['target']),rows(v))
    chosen=min(STRENGTHS,key=lambda z:(selection[str(z)]['patient_balanced_mean'],STRENGTHS.index(z)))
    np.savez(folder/'selection.npz',loglik=vl,**vp)
    el=vl if protocol=='mixed' else likelihood(model,ev['tokens']);cl=likelihood(model,ev['tokens'],collapse=True)
    tx=t['tokens'].astype(np.float64)[:,:,9:12].mean(1);ex=ev['tokens'].astype(np.float64)[:,:,9:12].mean(1);common=novelty(tx,ex)
    arrays={'target':ev['target'],'patient':ev['patient'],'site':ev['site'],'loglik':el,'collapse_loglik':cl,'common_risk':common};metrics={}
    for key,ll,strength in [(f't{s}',el,s) for s in STRENGTHS]+[('collapsed',cl,chosen)]:
        p,posterior=color_output(ll,model['palette'],strength)
        own=(posterior*delta_e00(p[:,None],model['palette'][None])).sum(1)
        m,e,order=score(p,common,ev);m.pop('predicted_error_mae_or_dispersion_mae');pm,_,po=score(p,own,ev)
        metrics[key]={'common':m,'posterior':pm};arrays[key]=p;arrays[key+'_error']=e;arrays[key+'_risk']=own
        arrays[key+'_curve']=np.cumsum(e[order])/np.arange(1,len(e)+1);arrays[key+'_posterior_curve']=np.cumsum(e[po])/np.arange(1,len(e)+1)
    np.savez(folder/'evaluation.npz',**arrays)
    r={'protocol':protocol,'representation':representation,'degree':degree,'components':components,'seed':seed,'selected_strength':chosen,
        'selection':selection,'metrics':metrics,'fit_seconds':fit_seconds,'stored_numeric_scalars':sum(a.size for a in model.values() if a.dtype.kind in 'fiu'),
        'model_bytes':(folder/'model.npz').stat().st_size,'model_sha256':sha(folder/'model.npz'),'selection_sha256':sha(folder/'selection.npz'),'evaluation_sha256':sha(folder/'evaluation.npz'),
        'train_people':len(set(t['patient'])),'selection_people':len(set(v['patient'])),'evaluation_people':len(set(ev['patient'])),
        'train_images':len(t['target']),'evaluation_images':len(ev['target']),'source_exploratory_only':True,'risk_calibrated':False,
        'reserved_endpoint_access':False,'protocol_sha256':sha(PROTOCOL),'training_code_sha256':sha(Path(__file__))}
    write(out/'result.json',r);print(json.dumps({'completed':name,'strength':chosen,'mean':metrics[f't{chosen}']['common']['full']['mean'],'collapsed':metrics['collapsed']['common']['full']['mean'],'seconds':fit_seconds}),flush=True)
```

</details>
