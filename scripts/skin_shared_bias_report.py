"""Aggregate all predeclared shared-bias fits, preserving ordinary comparators."""
import csv
import json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT, sha
from skin_shared_bias_train import OUT, RUN
from skin_shared_bias import ARMS


def main():
    audit = json.loads((OUT/'audit.json').read_bytes())
    assert audit['status'] == 'PASS' and audit['fits'] == 36
    paths = sorted(OUT.glob('*/result.json')); assert len(paths) == 36
    records = [json.loads(p.read_bytes()) for p in paths]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    groups = []
    def aggregate(rows, protocol, arm):
        rows = sorted(rows, key=lambda r: r['seed'])
        assert [r['seed'] for r in rows] == [17, 29, 43]
        return {'protocol': protocol, 'arm': arm, 'seed_means': [r['full']['mean'] for r in rows],
                'mean': float(np.mean([r['full']['mean'] for r in rows])),
                'p95': float(np.mean([r['full']['p95'] for r in rows]))}
    for protocol in ['mixed', 'from_SLR', 'from_ipod']:
        for arm in ARMS:
            rows = [r for r in records if r['protocol'] == protocol and r['objective'] == arm]
            g = aggregate(rows, protocol, arm)
            g['risk80'] = float(np.mean([r['uncalibrated_disagreement_coverage'][3]['mean'] for r in rows]))
            g['stored_parameters'] = rows[0]['stored_parameters']
            g['max_train_selection_mib'] = max(r['peak_train_allocated_mib'] for r in rows)
            groups.append(g)
        for oldarm in ['plain_mse', 'mixture_mse']:
            rows = []
            for seed in [17, 29, 43]:
                p = ROOT/'docs/benchmarks/skin_capture_v1'/f'{protocol}__{oldarm}__s{seed}'/'result.json'
                rows.append(json.loads(p.read_bytes())); bindings[str(p.relative_to(ROOT))] = sha(p)
            groups.append(aggregate(rows, protocol, 'historical_'+oldarm))
    with (OUT/'uncalibrated_risk_coverage.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protocol', 'objective', 'seed', 'accepted', 'population', 'coverage', 'mean_delta_e00'])
        for r in records:
            with np.load(RUN/f"{r['protocol']}__mixture_{r['objective']}__s{r['seed']}"/'evaluation.npz') as a:
                error = a['error'][np.argsort(a['risk'], kind='stable')]
                for k, v in enumerate(np.cumsum(error)/np.arange(1, len(error)+1), 1):
                    writer.writerow([r['protocol'], r['objective'], r['seed'], k, len(error), k/len(error), float(v)])
    comparisons = []
    for protocol, reference_arm in [('mixed','mixture_mse'),('from_SLR','mixture_mse'),('from_ipod','plain_mse')]:
        for arm in ARMS:
            paired = []; seed_deltas = []
            for seed in [17,29,43]:
                folder = RUN/f'{protocol}__mixture_{arm}__s{seed}'
                baseline = ROOT/'experiments/runs/skin_capture_v1'/f'{protocol}__{reference_arm}__s{seed}'/'evaluation.npz'
                with np.load(folder/'evaluation.npz') as a, np.load(baseline) as b:
                    np.testing.assert_array_equal(a['target'],b['target'])
                    np.testing.assert_array_equal(a['patient'],b['patient'])
                    difference = a['error']-b['error']; people=np.unique(a['patient'])
                    paired.append([float(difference[a['patient']==patient].mean()) for patient in people])
                    seed_deltas.append(float(difference.mean()))
            per_person=np.mean(paired,axis=0);rng=np.random.default_rng(1337)
            samples=per_person[rng.integers(0,len(per_person),(5000,len(per_person)))].mean(1)
            comparisons.append({'protocol':protocol,'arm':arm,'reference':reference_arm,
                'people':len(per_person),'image_mean_differences':seed_deltas,
                'seed_wins':sum(v<0 for v in seed_deltas),
                'person_balanced_mean_difference':float(per_person.mean()),
                'descriptive_patient_bootstrap_95':np.quantile(samples,[.025,.975]).tolist()})
    result = {'scope': 'SOURCE ONLY; three seed scores averaged, not an ensemble',
              'posthoc_descriptive_comparisons':comparisons,
              'groups': groups, 'result_bindings': bindings, 'audit_sha256': sha(OUT/'audit.json'),
              'report_script_sha256': sha(Path(__file__)), 'curve_sha256': sha(OUT/'uncalibrated_risk_coverage.csv')}
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    lines = ['# Shared skin-color bias versus paired-view agreement', '',
             'All 36 predeclared source fits retained. Native instrument Lab / CIEDE2000.',
             'Every reported model prediction uses one image. Values average three seed',
             'scores, not predictions. Source camera families have already been examined;',
             'this is neither fresh independent confirmation nor facial-phone accuracy.', '',
             '| Protocol | Objective | Mean DeltaE00 | p95 | Three seed means |',
             '|---|---|---:|---:|---|']
    for g in groups:
        lines.append(f"| {g['protocol']} | {g['arm']} | {g['mean']:.4f} | {g['p95']:.4f} | "+', '.join(f'{v:.4f}' for v in g['seed_means'])+' |')
    lines += ['', '| Protocol | Objective | Mean at 80% | Stored parameters | Peak fit+selection MiB |',
              '|---|---|---:|---:|---:|']
    for g in groups:
        if 'risk80' in g:
            lines.append(f"| {g['protocol']} | {g['arm']} | {g['risk80']:.4f} | {g['stored_parameters']} | {g['max_train_selection_mib']:.2f} |")
    lines += ['', 'Risk is uncalibrated hypothesis RMS Lab dispersion, not predicted DeltaE00.',
              'All six coverage levels are in run JSONs; [full curves](uncalibrated_risk_coverage.csv).',
              'No new latency, deployment, or camera-independent accuracy claim.', '',
              f"Audit: {audit['exact_prediction_array_replays']} exact color arrays, {audit['gate_and_hypothesis_replays']} gate/hypothesis replays,",
              f"{audit['independent_scalar_delta_e00_comparisons']} independent scalar color cases, {audit['independent_coverage_rows']} coverage rows,",
              f"{audit['identical_historical_control_final_weights']} exact historical control final weights.",
              'TRAIN-only scales and same-camera epoch selection checked. Reserved endpoints unused.', '',
              '[Protocol](../../research/skin_shared_bias_protocol_v1.md), [audit](audit.json),',
              '[decision](../../research/skin_correspondence_next_decision.md).']
    lines += ['', '## Descriptive comparison with strongest historical single-model control', '',
              'Post-hoc source comparisons, not confirmatory inference. Each person is one',
              'bootstrap cluster; seed errors are averaged within person before 5,000 resamples.',
              'Only 3 or 6 validation people, and adaptive source reuse, limit these intervals.', '',
              '| Protocol | Objective | Reference | Seed wins | Person mean difference | 95% descriptive interval |',
              '|---|---|---|---:|---:|---|']
    for c in comparisons:
        lo,hi=c['descriptive_patient_bootstrap_95']
        lines.append(f"| {c['protocol']} | {c['arm']} | {c['reference']} | {c['seed_wins']}/3 | {c['person_balanced_mean_difference']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    for g in groups: print(json.dumps(g))


if __name__ == '__main__':
    main()
