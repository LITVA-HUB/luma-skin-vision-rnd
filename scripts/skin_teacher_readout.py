"""Finite, source-only Ridge screen of licensed frozen teacher information."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load as load_pixels
from skin_teacher_features import OUT, CACHE, PROTOCOL, load as load_teacher
from skin_pair_train import subset, rows, write
from skin_mskcc_summary_pilot import summarize

RUN = ROOT/'experiments/runs/skin_teacher_readout_v1'
ARMS = ['color', 'teacher', 'combined', 'shuffle17', 'shuffle29', 'shuffle43']
ALPHAS = [.001, .01, .1, 1., 10., 100.]


def fit_blocks(blocks):
    result = []
    for value in blocks:
        x = np.asarray(value, dtype=np.float64)
        scale = x.std(0); scale = np.where(scale < 1e-8, 1., scale)
        result.append({'mean': x.mean(0), 'std': scale, 'width': x.shape[1]})
    return result


def transform_blocks(blocks, scalers):
    return np.concatenate([(np.asarray(x, dtype=np.float64)-s['mean'])/s['std']/np.sqrt(s['width'])
                           for x, s in zip(blocks, scalers, strict=True)], axis=1)


def site_weights(sites):
    _, index, counts = np.unique(sites, return_inverse=True, return_counts=True)
    value = 1./counts[index]
    return value/value.mean()


def source_permutation(n, seed, role):
    offsets = {'fit': 101, 'selection': 211, 'evaluation': 307}
    if role not in offsets:
        raise ValueError('Only fitting and source validation subsets may be shuffled')
    return np.random.default_rng(seed+offsets[role]).permutation(n)


def blocks(data, arm, role):
    if arm == 'color': return [data['color']]
    if arm == 'teacher': return [data['teacher']]
    if arm == 'combined': return [data['color'], data['teacher']]
    seed = int(arm.removeprefix('shuffle'))
    return [data['color'], data['teacher'][source_permutation(len(data['target']), seed, role)]]


def source_data():
    values = []
    for role in ['train', 'validation']:
        data = load_pixels(role); t = load_teacher(role)
        np.testing.assert_array_equal(data['image'], t['image'])
        data['teacher'] = t['teacher']; values.append(data)
    return values


def protocol_data(train, val, protocol):
    if protocol == 'mixed': return train, val, val
    device = protocol[5:]
    return subset(train, train['device'] == device), subset(val, val['device'] == device), subset(val, val['device'] != device)


def designs(train, selection, evaluation, arm, protocol):
    b = blocks(train, arm, 'fit'); scalers = fit_blocks(b)
    x = transform_blocks(b, scalers)
    sx = transform_blocks(blocks(selection, arm, 'selection'), scalers)
    ex = sx if protocol == 'mixed' else transform_blocks(blocks(evaluation, arm, 'evaluation'), scalers)
    return x, sx, ex, scalers


def main():
    audit = json.loads((OUT/'feature_audit.json').read_bytes())
    assert audit['status'] == 'PASS' and audit['exact_feature_vectors'] == 1230
    train, val = source_data(); RUN.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), PROTOCOL, ROOT/'tests/test_skin_teacher_readout.py',
             ROOT/'scripts/skin_teacher_features.py', OUT/'feature_receipt.json', OUT/'feature_audit.json',
             ROOT/'scripts/skin_pair_train.py', ROOT/'scripts/skin_mskcc_summary_pilot.py',
             ROOT/'src/luma_skin_vision/color.py']
    files += [CACHE/(role+'.npz') for role in ['train', 'validation']]
    write(OUT/'readout_lock.json', {'bindings': {str(p.relative_to(ROOT)): sha(p) for p in files},
        'arms': ARMS, 'alphas': ALPHAS, 'scope': 'SOURCE ONLY; pretraining overlap unknown'})
    with threadpool_limits(limits=4):
        for protocol in ['mixed', 'from_SLR', 'from_ipod']:
            tr, selection, evaluation = protocol_data(train, val, protocol)
            assert set(tr['patient']).isdisjoint(selection['patient']) and set(tr['patient']).isdisjoint(evaluation['patient'])
            if protocol != 'mixed':
                assert set(tr['device']).isdisjoint(evaluation['device'])
            ym = tr['target'].mean(0); ys = tr['target'].std(0); y = (tr['target']-ym)/ys
            weight = site_weights(tr['site'])
            for arm in ARMS:
                group = f'{protocol}__{arm}'; local = RUN/group; public = OUT/group
                local.mkdir(); public.mkdir()
                x, sx, ex, scalers = designs(tr, selection, evaluation, arm, protocol)
                risk = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(x).kneighbors(ex)[0].mean(1)
                order = np.argsort(risk, kind='stable'); candidates = []
                for alpha in ALPHAS:
                    name = f'a{alpha:g}'; folder = local/name; folder.mkdir()
                    model = Ridge(alpha=alpha, solver='svd').fit(x, y, sample_weight=weight)
                    sp = model.predict(sx)*ys+ym; p = model.predict(ex)*ys+ym
                    se = delta_e00(sp, selection['target']); e = delta_e00(p, evaluation['target'])
                    assert np.isfinite(p).all() and np.isfinite(sp).all()
                    np.savez(folder/'model.npz', coef=model.coef_, intercept=model.intercept_,
                        target_mean=ym, target_std=ys, x_mean=np.concatenate([s['mean'] for s in scalers]),
                        x_std=np.concatenate([s['std'] for s in scalers]), widths=np.array([s['width'] for s in scalers]),
                        fit_weight=weight)
                    np.savez(folder/'selection.npz', prediction=sp, target=selection['target'], error=se)
                    np.savez(folder/'evaluation.npz', prediction=p, target=evaluation['target'], error=e, risk=risk,
                             patient=evaluation['patient'], site=evaluation['site'])
                    record = {'protocol': protocol, 'arm': arm, 'alpha': alpha, 'features': x.shape[1],
                        'selection': summarize(se, rows(selection)), 'full': summarize(e, rows(evaluation)),
                        'readout_parameters': int(model.coef_.size+model.intercept_.size),
                        'model_sha256': sha(folder/'model.npz'), 'model_bytes': (folder/'model.npz').stat().st_size,
                        'selection_sha256': sha(folder/'selection.npz'), 'evaluation_sha256': sha(folder/'evaluation.npz'),
                        'coverage': [], 'strata': {}, 'reserved_endpoints_used': False}
                    for coverage in [1., .95, .9, .8, .7, .6]:
                        ix = order[:int(np.ceil(len(e)*coverage))]
                        record['coverage'].append({'coverage': coverage, **summarize(e[ix], rows(subset(evaluation, ix)))})
                    for key in ['device', 'image_type', 'mode']:
                        record['strata'][key] = {str(v): summarize(e[evaluation[key] == v], rows(subset(evaluation, evaluation[key] == v))) for v in np.unique(evaluation[key])}
                    write(public/(name+'.json'), record); candidates.append(record)
                best = min(candidates, key=lambda r: r['selection']['patient_balanced_mean'])
                write(public/'chosen.json', {'alpha': best['alpha'], 'selection_patient_mean': best['selection']['patient_balanced_mean'],
                    'result_sha256': sha(public/f"a{best['alpha']:g}.json"), 'readout_lock_sha256': sha(OUT/'readout_lock.json')})
                print(json.dumps({'completed': group, 'alpha': best['alpha'], 'mean': best['full']['mean'],
                                  'risk80': best['coverage'][3]['mean']}), flush=True)


if __name__ == '__main__':
    main()
