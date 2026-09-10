"""Auditable v1 loader repair: V2 checkpoints wrap tensors in `state`."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cc_phone_benchmark import (
    BENCH,
    CACHE,
    ROOT,
    check_cache,
    load,
    risk_table,
    scene_draws,
)
from cc_phone_benchmark import (
    checked_lock as original_checked_lock,
)

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json


def load_checkpoint(path, kind):
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["state"] if kind == "v2" else payload


def checked_lock(digest):
    lock = original_checked_lock(digest)
    repair = load(BENCH/"loader_repair_v1_1.json")
    if repair["original_lock_sha256"] != digest or sha256(Path(__file__)) != repair["repair_script_sha256"]:
        raise ValueError("Loader repair binding mismatch")
    return lock


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


def evaluate(digest):
    from cc_v5_report import reproduction_degrees

    from luma_skin_vision.cc.core import angular, summarize

    lock = checked_lock(digest)
    check_cache(digest)
    pdir = BENCH/'predictions_v1_1'
    bound = load(pdir/'manifest.json')
    if bound['method_lock_sha256'] != digest or bound['preparation_sha256'] != sha256(CACHE/'preparation.json'):
        raise ValueError('Wrong prediction bindings')
    for name, expected in bound['sha256'].items():
        if sha256(pdir/name) != expected:
            raise ValueError('Changed predictions')
    if (BENCH/'results.json').exists():
        raise ValueError('Existing evaluation; no overwrites')
    rows = load(CACHE/'reference_manifest.json')
    with np.load(CACHE/'references.npz', allow_pickle=False) as ref:
        gt, scorable = ref['gt'], ref['valid']
    cameras = np.array([r['camera'] for r in rows])
    records, all_values = [], {}
    for method in lock['methods']:
        with np.load(pdir/(method['id']+'.npz'), allow_pickle=False) as saved:
            values = {k:saved[k] for k in saved.files}
        if values['ids'].tolist() != [r['id'] for r in rows]:
            raise ValueError('Prediction row identity/order mismatch')
        error = np.full(len(rows), np.nan)
        error[scorable] = reproduction_degrees(values['pred'][scorable], gt[scorable])
        all_values[method['id']] = {**values,'error':error}
        for domain in ('all','samsung','oppo'):
            planned_mask = np.ones(len(rows), bool) if domain == 'all' else cameras == domain
            ix = np.where(planned_mask & scorable)[0]
            record = {'id':method['id'],'family':method['family'],'domain':domain,
                      'planned':int(planned_mask.sum()),'scorable':len(ix),
                      'model_refused':int((~values['valid'][ix]).sum()), 'reproduction':None,
                      'recovery':None,'selective':None,'source_thresholds':None}
            if len(ix):
                record['reproduction'] = summarize(error[ix])
                record['recovery'] = summarize(angular(values['pred'][ix],gt[ix]))
                if 'risk' in values:
                    record['selective'] = risk_table(error[ix],values['risk'][ix],values['valid'][ix],
                                                     [rows[i]['id'] for i in ix],int(planned_mask.sum()))
                if 'selection' in method:
                    cal = np.asarray(load(ROOT/method['selection'])['heads']['combined']['cal_scores'])
                    thresholds = {}
                    for c in (95,90,80,70,60):
                        threshold = float(np.quantile(cal,c/100))
                        accepted = values['valid'][ix] & (values['risk'][ix] <= threshold)
                        thresholds[str(c)] = {'threshold':threshold,'accepted':int(accepted.sum()),
                            'coverage_scorable':float(accepted.mean()),'coverage_planned':float(accepted.sum()/planned_mask.sum()),
                            'mean':float(error[ix][accepted].mean()) if accepted.any() else None}
                    record['source_thresholds'] = thresholds
            records.append(record)
    # Paired scene bootstrap, family metrics are means across seeds, never ensembles.
    draws = scene_draws([r['scene'] for r in rows])
    bootstrap = {}
    family_ids = {f:[m['id'] for m in lock['methods'] if m['family']==f] for f in ('v2_sog','v2_direct','gw_ridge1')}
    family_draw = {f:[] for f in family_ids}
    for draw in draws:
        ix = draw[scorable[draw]]
        if not len(ix):
            continue
        for family, ids in family_ids.items():
            per_model = []
            for name in ids:
                v = all_values[name]
                order = np.flatnonzero(v['valid'][ix])
                order = np.asarray(sorted(order, key=lambda i: (v['risk'][ix[i]],
                    hashlib.sha256(rows[ix[i]]['id'].encode()).hexdigest())), dtype=int)
                count = min(len(order),max(1,int(.8*len(ix))))
                per_model.append([v['error'][ix].mean(),v['error'][ix][order[:count]].mean() if count else np.nan])
            family_draw[family].append(np.mean(per_model,axis=0))
    for other in ('v2_direct','gw_ridge1'):
        delta = np.asarray(family_draw['v2_sog'])-np.asarray(family_draw[other])
        bootstrap[other] = {'difference':'SoG minus comparator; negative favors SoG',
                            'draws':len(delta),'full_mean_ci95':np.nanpercentile(delta[:,0],[2.5,97.5]).tolist(),
                            'risk80_ci95':np.nanpercentile(delta[:,1],[2.5,97.5]).tolist()}
    write_json(BENCH/'results.json', {'method_lock_sha256':digest,'preparation_sha256':sha256(CACHE/'preparation.json'),
               'predictions_manifest_sha256':sha256(pdir/'manifest.json'),
               'status':'MEASURED CUSTOM SOURCE-TO-PHONE TRANSFER; NOT SKIN/JPEG/HEIC VALIDATION',
               'rows':rows,'records':records,'paired_scene_bootstrap':bootstrap})
    print(json.dumps({'scorable':int(scorable.sum()),'planned':len(rows),'methods':len(lock['methods'])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("predict", "evaluate"))
    parser.add_argument("--lock-sha256", required=True)
    args = parser.parse_args()
    globals()[args.action](args.lock_sha256)
