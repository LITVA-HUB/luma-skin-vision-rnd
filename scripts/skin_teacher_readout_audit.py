"""Independent weighted-equation, scaler, metric and source-split audit."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from skin_mskcc_data import ROOT, sha
from skin_mskcc_audit import scalar_de
from skin_teacher_readout import OUT, RUN, ARMS, ALPHAS, source_data


def main():
    lock = json.loads((OUT/'readout_lock.json').read_bytes())
    for name, digest in lock['bindings'].items(): assert sha(ROOT/name) == digest, name
    tr, va = source_data(); records = sorted(OUT.glob('*__*/a*.json')); assert len(records) == 108
    replays = 0; cases = 0; coverage = 0; metric_gap = 0.; equation_gap = 0.; coefficient_gap = 0.; density_gap = 0.
    with threadpool_limits(limits=4):
        for protocol in ['mixed', 'from_SLR', 'from_ipod']:
            if protocol == 'mixed': datasets = [tr, va, va]
            else:
                c = protocol[5:]
                datasets = [{k: v[d['device'] == c if keep else d['device'] != c] for k, v in d.items()}
                            for d, keep in [(tr, True), (va, True), (va, False)]]
                assert set(datasets[0]['device']).isdisjoint(datasets[2]['device'])
            assert set(datasets[0]['patient']).isdisjoint(datasets[1]['patient'])
            assert set(datasets[0]['patient']).isdisjoint(datasets[2]['patient'])
            train = datasets[0]; w = np.array([1/sum(train['site'] == s) for s in train['site']]); w /= w.mean()
            ym = train['target'].mean(0); ys = train['target'].std(0); y = (train['target']-ym)/ys
            for arm in ARMS:
                input_blocks = []
                for j, data in enumerate(datasets):
                    color = data['color'].astype(np.float64); teacher = data['teacher'].astype(np.float64)
                    if arm.startswith('shuffle'):
                        offset = [101, 211, 211 if protocol == 'mixed' else 307][j]
                        permutation = np.random.default_rng(int(arm[7:])+offset).permutation(len(color))
                        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(color)))
                        teacher = teacher[permutation]
                    input_blocks.append([color] if arm == 'color' else [teacher] if arm == 'teacher' else [color, teacher])
                means = [b.mean(0) for b in input_blocks[0]]
                scales = [np.where(b.std(0) < 1e-8, 1., b.std(0)) for b in input_blocks[0]]
                widths = [b.shape[1] for b in input_blocks[0]]
                design = [np.concatenate([(b-m)/s/np.sqrt(d) for b, m, s, d in zip(bs, means, scales, widths, strict=True)], 1) for bs in input_blocks]
                x = design[0]; augmented = np.column_stack([x, np.ones(len(x))])
                gram = augmented.T@(w[:, None]*augmented); rhs = augmented.T@(w[:, None]*y)
                nearest = np.array([np.sort(np.linalg.norm(x-row, axis=1))[:5].mean() for row in design[2]])
                group = f'{protocol}__{arm}'; selection_scores = []
                for alpha in ALPHAS:
                    name = f'a{alpha:g}'; folder = RUN/group/name
                    record = json.loads((OUT/group/(name+'.json')).read_bytes())
                    for file, key in [('model.npz', 'model_sha256'), ('selection.npz', 'selection_sha256'), ('evaluation.npz', 'evaluation_sha256')]:
                        assert sha(folder/file) == record[key]
                    with np.load(folder/'model.npz') as model:
                        np.testing.assert_array_equal(model['target_mean'], ym); np.testing.assert_array_equal(model['target_std'], ys)
                        np.testing.assert_array_equal(model['x_mean'], np.concatenate(means)); np.testing.assert_array_equal(model['x_std'], np.concatenate(scales))
                        np.testing.assert_array_equal(model['widths'], widths); np.testing.assert_array_equal(model['fit_weight'], w)
                        coef = model['coef']; intercept = model['intercept']; beta = np.vstack([coef.T, intercept])
                        system = gram.copy(); system[np.arange(len(coef.T)), np.arange(len(coef.T))] += alpha
                        residual = np.linalg.norm(system@beta-rhs)/max(np.linalg.norm(rhs), 1.)
                        equation_gap = max(equation_gap, float(residual))
                        independently_solved = np.linalg.solve(system, rhs)
                        coefficient_gap = max(coefficient_gap, float(np.linalg.norm(independently_solved-beta)/max(np.linalg.norm(beta), 1.)))
                    for index, label in [(1, 'selection'), (2, 'evaluation')]:
                        with np.load(folder/(label+'.npz')) as saved:
                            prediction = (design[index]@coef.T+intercept)*ys+ym
                            np.testing.assert_array_equal(prediction, saved['prediction']); replays += 1
                            np.testing.assert_array_equal(saved['target'], datasets[index]['target'])
                            errors = np.array([scalar_de(p, q) for p, q in zip(prediction, saved['target'], strict=True)])
                            metric_gap = max(metric_gap, float(np.max(np.abs(errors-saved['error'])))); cases += len(errors)
                            expected = record['selection' if index == 1 else 'full']
                            person_mean = np.mean([errors[datasets[index]['patient'] == p].mean() for p in np.unique(datasets[index]['patient'])])
                            for key, value in [('mean', errors.mean()), ('median', np.median(errors)), ('p95', np.quantile(errors, .95)), ('patient_balanced_mean', person_mean)]:
                                metric_gap = max(metric_gap, abs(float(value)-expected[key]))
                            if index == 1:
                                selection_scores.append(float(person_mean))
                            else:
                                density_gap = max(density_gap, float(np.max(np.abs(nearest-saved['risk']))))
                                order = sorted(range(len(errors)), key=lambda i: (float(saved['risk'][i]), i))
                                for cr in record['coverage']:
                                    n = int(np.ceil(len(errors)*cr['coverage'])); assert n == cr['n']
                                    e = errors[order[:n]]
                                    for key, value in [('mean', e.mean()), ('median', np.median(e)), ('p95', np.quantile(e, .95)), ('above_5_fraction', (e > 5).mean()), ('above_10_fraction', (e > 10).mean())]:
                                        metric_gap = max(metric_gap, abs(float(value)-cr[key]))
                                    coverage += 1
                chosen = json.loads((OUT/group/'chosen.json').read_bytes())
                assert chosen['alpha'] == ALPHAS[int(np.argmin(selection_scores))]
                assert chosen['result_sha256'] == sha(OUT/group/f"a{chosen['alpha']:g}.json")
    assert metric_gap < 1e-10 and density_gap < 1e-10 and equation_gap < 1e-8 and coefficient_gap < 1e-7
    result = {'status': 'PASS', 'fits': 108, 'exact_prediction_arrays': replays,
              'independent_scalar_delta_e00_cases': cases, 'independent_coverage_rows': coverage,
              'maximum_metric_gap': metric_gap, 'maximum_density_gap': density_gap,
              'maximum_relative_normal_equation_residual': equation_gap,
              'maximum_relative_independent_solve_difference': coefficient_gap,
              'all_source_scalers_weights_shuffles_and_selections_checked': True,
              'reserved_endpoints_used': False, 'readout_lock_sha256': sha(OUT/'readout_lock.json'),
              'audit_script_sha256': sha(Path(__file__))}
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
