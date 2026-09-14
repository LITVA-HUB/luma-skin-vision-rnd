# `scripts/cc_phone_benchmark.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/cc_phone_benchmark.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Locked source-to-phone evaluation. No training and no old-run mutations.

SHA-256 исходника: `bc352731e7c3614a869f31cfac1855d775a6bd599d830aa1baca73b91bb4e102`. Строк: **404**.

## Зависимости

```python
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 12](../../../../scripts/cc_phone_benchmark.py#L12)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 13](../../../../scripts/cc_phone_benchmark.py#L13)

```python
PROV = ROOT/'docs/data/provenance/mobile_screen_2026_09_11'
```

[Строка 14](../../../../scripts/cc_phone_benchmark.py#L14)

```python
BENCH = ROOT/'docs/benchmarks/phone_v1'
```

[Строка 15](../../../../scripts/cc_phone_benchmark.py#L15)

```python
LOCK = BENCH/'method_lock.json'
```

[Строка 16](../../../../scripts/cc_phone_benchmark.py#L16)

```python
CACHE = ROOT/'data/processed/phone_v1'
```

[Строка 17](../../../../scripts/cc_phone_benchmark.py#L17)

```python
SELECTION_SHA = '86503dc8d2fdb247a57376c5a94ce5e0515003294b61286653b1e68099c0d38d'
```

[Строка 18](../../../../scripts/cc_phone_benchmark.py#L18)

```python
VERIFICATION_SHA = 'a346e735673a43320c6e2bffd5b93268544f77dae25e90f51963920ebbc078ac'
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `load` | FunctionDef | См. реализацию | [L21](../../../../scripts/cc_phone_benchmark.py#L21) |
| `relative` | FunctionDef | См. реализацию | [L25](../../../../scripts/cc_phone_benchmark.py#L25) |
| `verified_path` | FunctionDef | См. реализацию | [L29](../../../../scripts/cc_phone_benchmark.py#L29) |
| `risk_table` | FunctionDef | См. реализацию | [L39](../../../../scripts/cc_phone_benchmark.py#L39) |
| `scene_draws` | FunctionDef | См. реализацию | [L61](../../../../scripts/cc_phone_benchmark.py#L61) |
| `freeze` | FunctionDef | См. реализацию | [L69](../../../../scripts/cc_phone_benchmark.py#L69) |
| `checked_lock` | FunctionDef | См. реализацию | [L160](../../../../scripts/cc_phone_benchmark.py#L160) |
| `prepare` | FunctionDef | См. реализацию | [L170](../../../../scripts/cc_phone_benchmark.py#L170) |
| `check_cache` | FunctionDef | См. реализацию | [L233](../../../../scripts/cc_phone_benchmark.py#L233) |
| `predict` | FunctionDef | См. реализацию | [L243](../../../../scripts/cc_phone_benchmark.py#L243) |
| `evaluate` | FunctionDef | См. реализацию | [L312](../../../../scripts/cc_phone_benchmark.py#L312) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>predict · L243–309</summary>

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
    out = BENCH/'predictions'
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
                model.load_state_dict(torch.load(ROOT/method['weight'], map_location='cpu', weights_only=True))
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
                model.load_state_dict(torch.load(ROOT/method['weight'], map_location='cpu', weights_only=True))
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
