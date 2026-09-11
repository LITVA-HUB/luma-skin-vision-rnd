"""Source-only measurement/capture diagnostics; no fitting or new test access."""
import itertools
import json
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT, sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de

OUT = ROOT / 'docs/benchmarks/skin_correspondence_v1'
PROTOCOL = ROOT / 'docs/research/skin_correspondence_protocol_v1.md'


def unique_references(repetitions, sites, patients):
    reps = np.asarray(repetitions, dtype=np.float64)
    if reps.shape != (len(sites), 3, 3) or not np.isfinite(reps).all():
        raise ValueError('Invalid repetitions')
    result = []
    for site in np.unique(sites):
        idx = np.flatnonzero(sites == site)
        if len(set(patients[idx])) != 1:
            raise ValueError('Conflicting site patient')
        if not np.array_equal(reps[idx], np.broadcast_to(reps[idx[0]], reps[idx].shape)):
            raise ValueError('Conflicting site reference')
        result.append({'indices': idx, 'repetitions': reps[idx[0]], 'images': len(idx)})
    return result


def site_decomposition(prediction, target, sites):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape != (len(sites), 3):
        raise ValueError('Invalid prediction/target shapes')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Invalid prediction/target values')
    values = []
    for site in np.unique(sites):
        idx = np.flatnonzero(sites == site)
        if not np.array_equal(target[idx], np.broadcast_to(target[idx[0]], target[idx].shape)):
            raise ValueError('Conflicting site target')
        p = prediction[idx]; q = p.mean(0)
        values.append([len(idx), np.square(p-target[idx]).sum(1).mean(),
                       np.square(q-target[idx[0]]).sum(), np.square(p-q).sum(1).mean()])
    v = np.asarray(values)
    result = {}
    for prefix, weights in [('image', v[:, 0]), ('site', np.ones(len(v)))]:
        for k, name in enumerate(['mse_lab', 'shared_bias_squared', 'within_site_variance'], 1):
            result[prefix+'_'+name] = float(np.average(v[:, k], weights=weights))
        total = result[prefix+'_mse_lab']
        result[prefix+'_shared_fraction'] = result[prefix+'_shared_bias_squared']/total if total else None
    return result


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'p95': float(np.quantile(x, .95)), 'above_5_fraction': float((x > 5).mean())}


def balanced(values, data):
    result = stats(values)
    for unit in ['site', 'patient']:
        result[unit+'_balanced_mean'] = float(np.mean([np.mean(values[data[unit] == s]) for s in np.unique(data[unit])]))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    tr, va = load('train'), load('validation')
    assert set(tr['patient']).isdisjoint(va['patient'])
    # This diagnostic never reads the raw all-role endpoints or reserved caches.
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), PROTOCOL,
        ROOT/'tests/test_skin_correspondence.py', ROOT/'scripts/skin_mskcc_pixels.py',
        ROOT/'scripts/skin_mskcc_audit.py', ROOT/'src/luma_skin_vision/color.py',
        ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',
        ROOT/'data/processed/skin_mskcc_pixels_v1/validation.npz']}
    models = [('skin_capture_v1', 'plain_mse'), ('skin_capture_v1', 'mixture_mse'),
              ('skin_train_branch_v1', 'graph_always')]
    endpoints = []
    for family, arm in models:
        base = ROOT/'docs/benchmarks'/family
        for name, digest in json.loads((base/'source_lock.json').read_bytes())['bindings'].items():
            assert sha(ROOT/name) == digest, name
        for protocol, seed in itertools.product(['mixed', 'from_SLR', 'from_ipod'], [17, 29, 43]):
            name = f'{protocol}__{arm}__s{seed}'
            paths = [base/name/'result.json', ROOT/'experiments/runs'/family/name/'evaluation.npz']
            endpoints.append((family, arm, protocol, seed, paths))
            for p in paths:
                bindings[str(p.relative_to(ROOT))] = sha(p)
    (OUT/'source_lock.json').write_text(json.dumps({'bindings': bindings,
        'reserved_endpoints_loaded': False}, indent=2)+'\n', encoding='utf8')
    measurement = []; audit_gap = 0.; scalar_cases = 0
    def checked(a, b):
        nonlocal audit_gap, scalar_cases
        a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
        values = delta_e00(a, b)
        independent = np.array([scalar_de(p, q) for p, q in zip(a.reshape(-1, 3), b.reshape(-1, 3), strict=True)]).reshape(values.shape)
        audit_gap = max(audit_gap, float(np.max(np.abs(values-independent))))
        scalar_cases += values.size
        return values
    train_sites = unique_references(tr['repetitions'], tr['site'], tr['patient'])
    light_edges = np.quantile([r['repetitions'].mean(0)[0] for r in train_sites], [.25, .5, .75])
    for role, data in [('train', tr), ('validation', va)]:
        refs = unique_references(data['repetitions'], data['site'], data['patient'])
        pair, center, loo = [], [], []
        for ref in refs:
            r = ref['repetitions']; q = r.mean(0)
            pair.append(checked(r[[0, 0, 1]], r[[1, 2, 2]]))
            center.append(checked(r, q))
            loo.append(checked(r, (r.sum(0)-r)/2))
        pair, center, loo = map(np.asarray, (pair, center, loo))
        first = np.array([r['indices'][0] for r in refs])
        devices = data['device'][first]; labs = data['target'][first]
        strata = {}
        for name, labels in [('device', devices), ('train_lightness_quartile', np.searchsorted(light_edges, labs[:, 0], side='right').astype(str))]:
            strata[name] = {str(label): {'sites': int(sum(labels == label)),
                'pairwise': stats(pair[labels == label].ravel()),
                'to_mean': stats(center[labels == label].ravel())} for label in np.unique(labels)}
        measurement.append({'role': role, 'images': len(data['target']), 'sites': len(refs),
            'people': len(set(data['patient'])), 'reference_copies_exact': True,
            'pairwise': stats(pair.ravel()), 'to_mean': stats(center.ravel()),
            'leave_one_reading_out': stats(loo.ravel()), 'strata': strata})
    records = []
    for family, arm, protocol, seed, paths in endpoints:
        mask = np.ones(len(va['target']), dtype=bool) if protocol == 'mixed' else va['device'] != protocol[5:]
        data = {k: v[mask] for k, v in va.items()}
        record = json.loads(paths[0].read_bytes())
        with np.load(paths[1]) as saved:
            p = saved['prediction'].astype(np.float64)
            np.testing.assert_array_equal(saved['target'], data['target'])
            e = checked(p, data['target'])
            np.testing.assert_allclose(e, saved['error'], atol=1e-10, rtol=0)
            assert abs(e.mean()-record['full']['mean']) < 1e-10
        each = checked(p[:, None, :], data['repetitions'])
        pair_means, mean_prediction_error = [], []
        for site in np.unique(data['site']):
            idx = np.flatnonzero(data['site'] == site)
            pairs = list(itertools.combinations(idx, 2))
            if pairs:
                i, j = np.asarray(pairs).T
                pair_means.append(float(checked(p[i], p[j]).mean()))
            mean_prediction_error.append(float(checked(p[idx].mean(0), data['target'][idx[0]])))
        strata = {}
        for key, labels in [('device', data['device']), ('mode', data['mode']), ('image_type', data['image_type']),
                            ('train_lightness_quartile', np.searchsorted(light_edges, data['target'][:, 0], side='right').astype(str))]:
            strata[key] = {str(label): stats(e[labels == label]) for label in np.unique(labels)}
        decomposition = site_decomposition(p, data['target'], data['site'])
        for unit in ['image', 'site']:
            assert abs(decomposition[unit+'_mse_lab'] - decomposition[unit+'_shared_bias_squared'] - decomposition[unit+'_within_site_variance']) < 1e-10
        records.append({'family': family, 'arm': arm, 'protocol': protocol, 'seed': seed,
            'single_image_error': balanced(e, data),
            'best_repeat_reference_error': stats(each.min(1)), 'worst_repeat_reference_error': stats(each.max(1)),
            'error_above5_for_all_three_readings_fraction': float((each.min(1) > 5).mean()),
            'site_balanced_capture_disagreement': stats(pair_means),
            'diagnostic_multiple_view_mean_error': stats(mean_prediction_error),
            'lab_squared_decomposition_not_delta_e00': decomposition, 'strata': strata})
    assert audit_gap < 1e-10
    result = {'scope': 'SOURCE DIAGNOSTIC; no model fitting; no independent test',
        'source_lock_sha256': sha(OUT/'source_lock.json'), 'train_site_lightness_quartile_edges': light_edges.tolist(),
        'measurement': measurement, 'models': records,
        'audit': {'independent_scalar_cases': scalar_cases, 'maximum_gap': audit_gap,
                  'reference_identity_and_patient_join_checked': True, 'mse_decomposition_identity_checked': True}}
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'measurement': measurement, 'audit': result['audit']}))


if __name__ == '__main__':
    main()
