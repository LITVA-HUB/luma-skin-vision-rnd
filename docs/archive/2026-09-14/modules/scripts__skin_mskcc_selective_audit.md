# `scripts/skin_mskcc_selective_audit.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_mskcc_selective_audit.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Read-only OOF or independent-test audit; cannot fit or alter any model.

SHA-256 исходника: `a159952e448aa01c232feffcae2cff438b9bc8d41c671a4de5ff9cfc1aae1af0`. Строк: **76**.

## Зависимости

```python
import argparse
import json
import joblib
import numpy as np
import torch
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_selective_core import OUT,RUN,patient_folds,plain_predict,verify_lock,deployment_designs
from skin_mskcc_selective_data import sealed
from skin_mskcc_audit import scalar_de
from skin_mskcc_color_registry import all_predictions
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `oof_audit` | FunctionDef | См. реализацию | [L15](../../../../scripts/skin_mskcc_selective_audit.py#L15) |
| `test_audit` | FunctionDef | См. реализацию | [L34](../../../../scripts/skin_mskcc_selective_audit.py#L34) |

## Все тестовые определения (1)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_audit` · L34

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_audit():
    torch.set_num_threads(4);lock_path=OUT/'final_lock.json';digest=sha(lock_path);lock=verify_lock(lock_path,digest,'final')
    data=sealed('test',lock_path,digest);record=json.loads((OUT/'test_results.json').read_bytes())
    assert record['final_lock_sha256']==digest
    maximum=0.;curves=0;arrays=0
    for path in sorted((RUN/'test').glob('*.npz')):
        d=np.load(path);np.testing.assert_array_equal(d['target'],data['target'])
        error=np.array([scalar_de(p,q) for p,q in zip(d['prediction'],d['target'])])
        maximum=max(maximum,float(np.max(np.abs(error-d['error']))));arrays+=1
        name=path.stem
        if name.startswith('color_'):
            reported=record['color_comparators'][name[6:]];score=d['density']
        else:reported=record['risk_models'][name];score=d['raw']
        order=sorted(range(len(error)),key=lambda i:(float(score[i]),str(data['image'][i])))
        for entry in reported['coverage']:
            n=int(np.ceil(entry['coverage']*len(error)));e=sorted(float(error[i]) for i in order[:n])
            mean=sum(e)/n;median=e[n//2] if n%2 else (e[n//2-1]+e[n//2])/2
            maximum=max(maximum,abs(mean-entry['mean']),abs(median-entry['median']))
            for q,key in [(.9,'p90'),(.95,'p95')]:
                z=(n-1)*q;l=int(z);value=e[l]+(e[min(l+1,n-1)]-e[l])*(z-l);maximum=max(maximum,abs(value-entry[key]))
            curves+=1
    # Independently replay all selected head+calibration objects.
    design=deployment_designs(data,load('train'));head_gap=0.
    for key,choice in lock['selected'].items():
        version,arm=key.split('__');d=np.load(RUN/'test'/(key+'.npz'))
        raw=np.maximum(joblib.load(ROOT/choice['path']).predict(design[version][arm]),0)
        cal=joblib.load(ROOT/lock['calibrators'][key]['path']).predict(raw)
        head_gap=max(head_gap,float(np.max(np.abs(raw-d['raw']))),float(np.max(np.abs(cal-d['calibrated']))))
    color_gap=0.;color_replays=all_predictions(data)
    for name,prediction in color_replays.items():
        saved=np.load(RUN/'test'/('color_'+name+'.npz'))
        color_gap=max(color_gap,float(np.max(np.abs(prediction-saved['prediction']))))
    assert maximum<1e-10 and head_gap==0 and color_gap==0
    result={'status':'PASS','test_prediction_arrays':arrays,'independent_coverage_cases':curves,
            'max_scalar_metric_gap':maximum,'head_and_calibrator_replays':len(lock['selected']),'max_head_replay_gap':head_gap,
            'color_system_replays':len(color_replays),'max_color_replay_gap':color_gap,
            'final_lock_sha256':digest,'test_results_sha256':sha(OUT/'test_results.json')}
    (OUT/'test_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
```

</details>
