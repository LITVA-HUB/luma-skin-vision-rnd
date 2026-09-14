# `scripts/cc_phone_alias.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_alias.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Additive dataset-key repair; preserves all original phone evidence.

SHA-256 исходника: `c6596f90c8730106dee8a0a8ddbc3ac80e7dc9bb0d42fe580ddad4b460a63d3c`. Строк: **276**.

## Зависимости

```python
import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np
from cc_phone_benchmark import (
    PROV,
    ROOT,
    load,
    risk_table,
    scene_draws,
    verified_path,
)
from cc_phone_benchmark import (
    checked_lock as original_checked_lock,
)
from cc_phone_benchmark_v1_1 import load_checkpoint
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 26](../../../../scripts/cc_phone_alias.py#L26)

```python
BENCH = ROOT/"docs/benchmarks/phone_v1_alias"
```

[Строка 27](../../../../scripts/cc_phone_alias.py#L27)

```python
CACHE = ROOT/"data/processed/phone_v1_alias"
```

[Строка 28](../../../../scripts/cc_phone_alias.py#L28)

```python
ORIGINAL = ROOT/"data/processed/phone_v1"
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `open_camera_rgb` | FunctionDef | См. реализацию | [L32](../../../../scripts/cc_phone_alias.py#L32) |
| `checked_lock` | FunctionDef | См. реализацию | [L43](../../../../scripts/cc_phone_alias.py#L43) |
| `check_cache` | FunctionDef | См. реализацию | [L54](../../../../scripts/cc_phone_alias.py#L54) |
| `prepare` | FunctionDef | См. реализацию | [L64](../../../../scripts/cc_phone_alias.py#L64) |
| `predict` | FunctionDef | См. реализацию | [L118](../../../../scripts/cc_phone_alias.py#L118) |
| `evaluate` | FunctionDef | См. реализацию | [L187](../../../../scripts/cc_phone_alias.py#L187) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L118–184</summary>

```python
def predict(digest):
    import joblib
    import torch
    from cc_v2_select import features, raw_predict
    from cc_v2_statistics import predict_arrays
    from cc_v5_model import CorrectionEvidenceNet
    from threadpoolctl import threadpool_limits

    from luma_skin_vision.cc.core import angular
    from luma_skin_vision.cc.v2 import CompactResidualCC, risk_features_invariant

    lock = checked_lock(digest)
    # Byte hashes are verified; reference values are never deserialized here.
    check_cache(digest)
    rows = load(CACHE/'input_manifest.json')
    with np.load(CACHE/'inputs.npz', allow_pickle=False) as inp:
        images, classical, input_valid = inp['images'], inp['experts'], inp['valid']
    out = BENCH/'predictions_v1_1'
    if out.exists():
        raise ValueError('Prediction directory already exists')
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    for method in lock['methods']:
        kind = method['kind']
        scores = None
        with threadpool_limits(limits=2), torch.no_grad():
            if kind == 'classical':
                pred = classical[:, method['expert_index']]
                scores = np.mean([angular(pred, classical[:, i]) for i in range(4)], axis=0)
                values = {'pred': pred, 'valid': input_valid.copy()}
            elif kind == 'statistics':
                values = predict_arrays(joblib.load(ROOT/method['weight']), images)
            elif kind == 'v2':
                model = CompactResidualCC(method['mode'], method['backbone']).eval()
                model.load_state_dict(load_checkpoint(ROOT/method['weight'], kind))
                values = {k:[] for k in ('pred','context','cheap_features','valid')}
                for start in range(0, len(images), 16):
                    x = torch.from_numpy(images[start:start+16])
                    pred, context = model(x)
                    feature = risk_features_invariant(x, pred, context)
                    for k, v in [('pred',pred),('context',context),('cheap_features',feature['cheap']),('valid',feature['valid'])]:
                        values[k].append(v.numpy())
                values = {k:np.concatenate(v) for k,v in values.items()}
            else:
                arm = method['arm']
                model = CorrectionEvidenceNet(mode='posterior' if arm == 'point' else arm.split('_')[0]).eval()
                model.load_state_dict(load_checkpoint(ROOT/method['weight'], kind))
                predictions, risk, valid = [], [], []
                for start in range(0, len(images), 16):
                    output = model(torch.from_numpy(images[start:start+16]))
                    predictions.append(output['base_pred' if arm == 'point' else 'pred'].numpy())
                    risk.append(output['risk'].numpy())
                    valid.append(output['valid'].numpy())
                values = {'pred':np.concatenate(predictions),'valid':np.concatenate(valid)}
                if arm != 'point':
                    scores = np.concatenate(risk)
            if kind in ('v2','statistics'):
                head = joblib.load(ROOT/method['head'])
                scores = raw_predict(head['model'], features(values, 'combined'))*head['scale']
            values['valid'] &= input_valid
            payload = {'pred':values['pred'], 'valid':values['valid'], 'ids':np.array([r['id'] for r in rows])}
            if scores is not None:
                payload['risk'] = scores
            np.savez_compressed(out/(method['id']+'.npz'), **payload)
        print(json.dumps({'predicted':method['id'],'gt_values_read':False}),flush=True)
    write_json(out/'manifest.json', {'method_lock_sha256':digest, 'preparation_sha256':sha256(CACHE/'preparation.json'),
               'gt_values_read':False, 'sha256':{p.name:sha256(p) for p in out.iterdir() if p.is_file()}})
```

</details>
