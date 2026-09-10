"""Locked source-to-phone evaluation. No training and no old-run mutations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT/'docs/data/provenance/mobile_screen_2026_09_11'
BENCH = ROOT/'docs/benchmarks/phone_v1'
LOCK = BENCH/'method_lock.json'
CACHE = ROOT/'data/processed/phone_v1'
SELECTION_SHA = '86503dc8d2fdb247a57376c5a94ce5e0515003294b61286653b1e68099c0d38d'
VERIFICATION_SHA = 'a346e735673a43320c6e2bffd5b93268544f77dae25e90f51963920ebbc078ac'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verified_path(root, name, index):
    path = (Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path outside dataset root')
    key = Path(name).as_posix()
    if key not in index or sha256(path) != index[key]['sha256']:
        raise ValueError('Acquired file digest mismatch: '+key)
    return path


def risk_table(error, score, valid, ids, planned):
    error, score, valid = np.asarray(error), np.asarray(score), np.asarray(valid, dtype=bool)
    if error.shape != score.shape or error.shape != valid.shape or len(ids) != len(error):
        raise ValueError('Unaligned risk arrays')
    if not np.isfinite(error).all() or not np.isfinite(score[valid]).all():
        raise ValueError('Nonfinite risk inputs')
    order = sorted(np.where(valid)[0], key=lambda i: (score[i], hashlib.sha256(ids[i].encode()).hexdigest()))
    curve = (np.cumsum(error[order])/np.arange(1, len(order)+1)).tolist()
    fixed = {}
    for c in (100, 95, 90, 80, 70, 60):
        count = min(len(order), max(1, int(len(error)*c/100))) if len(error) else 0
        accepted = error[order[:count]]
        fixed[str(c)] = {'accepted': count, 'coverage_scorable': count/len(error) if len(error) else 0,
                         'coverage_planned': count/planned if planned else 0,
                         'mean': float(accepted.mean()) if count else None,
                         'p95': float(np.percentile(accepted, 95)) if count else None,
                         'over10': int((accepted > 10).sum())}
    return {'fixed': fixed, 'order': list(map(int, order)), 'curve': curve,
            'coverage_scorable': (np.arange(1, len(order)+1)/max(1, len(error))).tolist(),
            'coverage_planned': (np.arange(1, len(order)+1)/max(1, planned)).tolist()}


def scene_draws(groups, repeats=2000, seed=20260911):
    groups = np.asarray(groups)
    unique = sorted(set(groups))
    members = {k: np.where(groups == k)[0] for k in unique}
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[k] for k in rng.choice(unique, len(unique), replace=True)]) for _ in range(repeats)]


def freeze():
    if LOCK.exists():
        raise ValueError('Method lock already exists')
    if sha256(PROV/'beyond_rgb_selection.json') != SELECTION_SHA or sha256(PROV/'verification_all.json') != VERIFICATION_SHA:
        raise ValueError('Original phone data bindings changed')
    frozen, methods = {}, []

    def bind(path):
        path = Path(path)
        frozen[relative(path)] = sha256(path)
        return relative(path)

    old_lock = load(ROOT/'docs/benchmarks/cc_v2/final_head_lock.json')
    bind(ROOT/'docs/benchmarks/cc_v2/final_head_lock.json')
    for mode in ('direct', 'sog'):
        for seed in (17, 29, 43):
            name = f'ccv2_{mode}_large_g0_s{seed}'
            run = ROOT/'experiments/runs'/name
            config, checkpoint = load(run/'config.json'), load(run/'checkpoint_manifest.json')
            if sha256(run/'model.pt') != checkpoint['checkpoint_sha256'] or sha256(run/'config.json') != checkpoint['config_sha256']:
                raise ValueError('V2 checkpoint/config identity changed')
            for src, expected in config['source_snapshot']['files'].items():
                if sha256(run/'source_snapshot'/src) != expected:
                    raise ValueError('V2 source snapshot changed')
                if src.endswith('.py'):
                    if sha256(ROOT/src) != expected:
                        raise ValueError('Live numerical Python differs from V2: '+src)
                    bind(ROOT/src)
            selection = run/'risk_v2/selection.json'
            state = load(selection)
            if sha256(selection) != old_lock['cnn_selectors'][name]:
                raise ValueError('V2 selector differs from original head lock')
            head = run/'risk_v2/combined.joblib'
            if sha256(head) != state['heads']['combined']['artifact_sha256']:
                raise ValueError('V2 calibrated head changed')
            bind(run/'checkpoint_manifest.json')
            methods.append({'id': name, 'kind': 'v2', 'family': 'v2_'+mode, 'seed': seed,
                            'mode': mode, 'backbone': 'large', 'weight': bind(run/'model.pt'),
                            'config': bind(run/'config.json'), 'selection': bind(selection), 'head': bind(head)})
    stats_run = ROOT/'experiments/runs/ccv2_statistics_fixed'
    screen = load(stats_run/'screen.json')
    bind(stats_run/'screen.json')
    if sha256(ROOT/'scripts/cc_v2_statistics.py') != screen['script_sha256']:
        raise ValueError('Statistics numerical implementation changed')
    for candidate in ('gw_ridge1', 'direct_hgb7'):
        record = next(r for r in screen['candidates'] if r['candidate'] == candidate)
        weight = stats_run/record['model_file']
        folder = ROOT/'experiments/runs/ccv2_statistics_eval'/candidate
        state = load(folder/'selection.json')
        if sha256(weight) != record['model_sha256'] or sha256(folder/'selection.json') != old_lock['statistics_selectors'][candidate]:
            raise ValueError('Statistics original estimator/head lock mismatch')
        if sha256(folder/'combined.joblib') != state['heads']['combined']['artifact_sha256']:
            raise ValueError('Statistics calibrated head changed')
        methods.append({'id': candidate, 'kind': 'statistics', 'family': candidate,
                        'weight': bind(weight), 'selection': bind(folder/'selection.json'),
                        'head': bind(folder/'combined.joblib')})
    for seed in (17, 29, 43):
        run = ROOT/f'experiments/runs/ccv5_paired_s{seed}'
        config = load(run/'config.json')
        manifest = load(run/'artifact_manifest.json')['sha256']
        bind(run/'config.json')
        bind(run/'artifact_manifest.json')
        for name, expected in config['scripts_sha256'].items():
            if sha256(ROOT/'scripts'/name) != expected:
                raise ValueError('V5 numerical code changed')
            bind(ROOT/'scripts'/name)
        for arm in config['arms']:
            path = run/arm/'best.pt'
            if sha256(path) != manifest[f'{arm}/best.pt']:
                raise ValueError('V5 best checkpoint changed')
            methods.append({'id': f'v5_{arm}_s{seed}', 'family': 'v5_'+arm, 'seed': seed,
                            'kind': 'v5', 'arm': arm, 'weight': bind(path), 'steps': 2})
    for i, name in enumerate(('gray_world', 'max_rgb', 'shades_gray', 'gray_edge')):
        methods.append({'id': name, 'family': name, 'kind': 'classical', 'expert_index': i})
    for name in ('cc_phone_benchmark.py', 'cc_phone_prepare.py', 'cc_phone_loader_audit.py',
                 'cc_v2_statistics.py', 'cc_v2_select.py', 'cc_v5_report.py'):
        bind(ROOT/'scripts'/name)
    for name in ('phone_reference_protocol_v1.md', 'phone_benchmark_protocol_v1.md'):
        bind(ROOT/'docs/research'/name)
    bind(PROV/'beyond_rgb_selection.json')
    bind(PROV/'verification_all.json')
    if len(methods) != 30 or len({m['id'] for m in methods}) != 30:
        raise ValueError('Expected 30 unique methods')
    write_json(LOCK, {'status': 'FROZEN BEFORE RESERVED PHONE PIXEL/GT DECODING',
                     'methods': methods, 'sha256': frozen, 'planned_inputs': 88,
                     'primary': 'v2_sog versus v2_direct and gw_ridge1; all seed outcomes retained',
                     'source_differences': 'Dependency declarations add HDF5 support; all V2 numerical Python matches original snapshot'})
    print(json.dumps({'lock_sha256': sha256(LOCK), 'methods': len(methods)}))


def checked_lock(digest):
    if sha256(LOCK) != digest:
        raise ValueError('Explicit method lock digest mismatch')
    lock = load(LOCK)
    for name, expected in lock['sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Locked artifact changed: '+name)
    return lock


def prepare(digest):
    from cc_phone_loader_audit import open_camera_rgb, patch_stats
    from cc_phone_prepare import reference_from_patches

    from luma_skin_vision.cc.core import experts
    from luma_skin_vision.cc.data import sample

    checked_lock(digest)
    if CACHE.exists():
        raise ValueError('Existing phone cache; never overwrite')
    selection = load(PROV/'beyond_rgb_selection.json')
    scenes = [s for s in selection['scenes'] if s['role'] == 'reserved_test']
    if len(scenes) != 44:
        raise ValueError('Wrong reserved scene count')
    verification = load(PROV/'verification_all.json')
    index = {Path(r['path']).as_posix(): r for r in verification['records']}
    data_root = ROOT/'data/public/beyond_rgb_phone'
    rows, images, targets, estimates = [], [], [], []
    CACHE.mkdir(parents=True)
    for scene in scenes:
        for camera in ('samsung', 'oppo'):
            prefix = f"beyond-unzip/beyondRGB/{scene['scene']}"
            paths = {key: verified_path(data_root, f'{prefix}/{part}/{camera}{suffix}', index)
                     for key, part, suffix in [('nt','NT','.h5'), ('wt','WT','.h5'), ('patch','WT','_cc_detection.json')]}
            ref = {'valid': False, 'gt': None, 'reason': 'Reference unavailable'}
            try:
                detection = load(paths['patch'])
                with open_camera_rgb(paths['wt'], camera) as data:
                    patches = {str(i): patch_stats(data, detection[f'patch_{i}']['corners']) for i in range(1, 25)}
                ref = reference_from_patches(patches)
            except (ValueError, KeyError, OSError) as exc:
                ref['reason'] = type(exc).__name__+': '+str(exc)
            input_valid, input_reason = True, None
            try:
                with open_camera_rgb(paths['nt'], camera) as data:
                    rgb = np.array(data, dtype=np.float32)
                if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                    raise ValueError('Released RGB is outside finite[0,1]')
                invalid = (rgb.max(-1) >= 254/255) | (rgb.max(-1) <= 0)
                rgb[invalid] = 0
                hypothesis, image = experts(rgb), sample(rgb, 128)
                masked_fraction = float(invalid.mean())
            except (ValueError, OSError) as exc:
                input_valid, input_reason = False, type(exc).__name__+': '+str(exc)
                image, hypothesis, masked_fraction = np.zeros((3,128,128), np.float32), np.ones((4,3))/np.sqrt(3), 1.
            rows.append({'id': f"beyond_rgb:{scene['scene']}:{camera}", 'scene': scene['scene'], 'camera': camera,
                         'reference': ref, 'input_valid': input_valid, 'input_reason': input_reason,
                         'masked_fraction': masked_fraction,
                         'privacy_mask_paths': [m['path'] for m in scene['members'] if 'blurred_areas' in m['path'] and camera in m['path']]})
            images.append(image)
            estimates.append(hypothesis)
            targets.append(ref['gt'] if ref['valid'] else [np.nan]*3)
            print(json.dumps({'prepared': len(rows), 'total': 88, 'reference_valid': ref['valid']}), flush=True)
    np.savez_compressed(CACHE/'inputs.npz', images=np.array(images), experts=np.array(estimates),
                        valid=np.array([r['input_valid'] for r in rows]))
    np.savez_compressed(CACHE/'references.npz', gt=np.array(targets), valid=np.array([r['reference']['valid'] for r in rows]))
    write_json(CACHE/'input_manifest.json', [{k:r[k] for k in ('id','scene','camera','input_valid')} for r in rows])
    write_json(CACHE/'reference_manifest.json', rows)
    write_json(CACHE/'preparation.json', {'method_lock_sha256': digest, 'planned': 88,
               'reference_valid': sum(r['reference']['valid'] for r in rows),
               'sha256': {p.name: sha256(p) for p in CACHE.iterdir() if p.is_file()}})


def check_cache(digest):
    receipt = load(CACHE/'preparation.json')
    if receipt['method_lock_sha256'] != digest:
        raise ValueError('Wrong preparation lock')
    for name, expected in receipt['sha256'].items():
        if sha256(CACHE/name) != expected:
            raise ValueError('Prepared cache changed')
    return receipt


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


def evaluate(digest):
    from cc_v5_report import reproduction_degrees

    from luma_skin_vision.cc.core import angular, summarize

    lock = checked_lock(digest)
    check_cache(digest)
    pdir = BENCH/'predictions'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','prepare','predict','evaluate'))
    parser.add_argument('--lock-sha256')
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        globals()[args.action](args.lock_sha256)
