"""All chosen source readouts and all candidate risk curves, without test claims."""
import csv
import json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT, sha
from skin_teacher_readout import OUT, RUN, ARMS, ALPHAS


def main():
    audit = json.loads((OUT/'audit.json').read_bytes())
    assert audit['status'] == 'PASS' and audit['fits'] == 108
    features = json.loads((OUT/'feature_receipt.json').read_bytes())
    bindings = {}; selected = []; historical = []
    with (OUT/'all_candidate_risk_coverage.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['protocol', 'arm', 'alpha', 'selected', 'accepted', 'population', 'coverage', 'mean_delta_e00'])
        for protocol in ['mixed', 'from_SLR', 'from_ipod']:
            for arm in ARMS:
                group = f'{protocol}__{arm}'; choice = json.loads((OUT/group/'chosen.json').read_bytes())
                for alpha in ALPHAS:
                    path = OUT/group/f'a{alpha:g}.json'; r = json.loads(path.read_bytes())
                    bindings[str(path.relative_to(ROOT))] = sha(path)
                    chosen = alpha == choice['alpha']
                    if chosen:
                        assert sha(path) == choice['result_sha256']
                        selected.append(r)
                    with np.load(RUN/group/f'a{alpha:g}'/'evaluation.npz') as saved:
                        e = saved['error'][np.argsort(saved['risk'], kind='stable')]
                        for k, value in enumerate(np.cumsum(e)/np.arange(1, len(e)+1), 1):
                            writer.writerow([protocol, arm, alpha, chosen, k, len(e), k/len(e), float(value)])
            for family, arm in [('skin_capture_v1', 'plain_mse'), ('skin_capture_v1', 'mixture_mse'), ('skin_train_branch_v1', 'graph_always')]:
                records = []
                for seed in [17, 29, 43]:
                    path = ROOT/'docs/benchmarks'/family/f'{protocol}__{arm}__s{seed}'/'result.json'
                    records.append(json.loads(path.read_bytes())); bindings[str(path.relative_to(ROOT))] = sha(path)
                historical.append({'protocol': protocol, 'method': arm,
                    'seed_means': [r['full']['mean'] for r in records],
                    'mean': float(np.mean([r['full']['mean'] for r in records]))})
    result = {'scope': 'SOURCE READOUT SCREEN; no deployable compact model or independent result',
              'selected': selected, 'historical': historical, 'result_bindings': bindings,
              'audit_sha256': sha(OUT/'audit.json'), 'feature_receipt_sha256': sha(OUT/'feature_receipt.json'),
              'report_script_sha256': sha(Path(__file__)), 'curve_sha256': sha(OUT/'all_candidate_risk_coverage.csv')}
    (OUT/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    lines = ['# Frozen teacher information for actual skin-color prediction', '',
             '108 locally reproduced Ridge fits: six arms, six alphas, three source protocols.',
             'Alpha chosen on same-camera source validation; all candidates retained. These',
             'source camera families and validation people have already been explored.',
             'Native instrument Lab / CIEDE2000, single-image predictions. No TEST/CAL use.', '',
             '| Protocol | Features | Selected alpha | Mean DeltaE00 | Median | p95 | Mean at 80% |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['protocol']} | {r['arm']} | {r['alpha']:g} | {r['full']['mean']:.4f} | {r['full']['median']:.4f} | {r['full']['p95']:.4f} | {r['coverage'][3]['mean']:.4f} |")
    lines += ['', 'Shuffled controls independently permute teacher features within fitting and',
              'source evaluation subsets, retaining the corresponding absolute-color features.',
              'Permutation seeds are null controls, not extra independent people or training',
              'replications. Readouts use a fixed SVD solution; there is no stochastic fit seed.', '',
              '| Protocol | Strong historical compact model | Mean over three seed scores |',
              '|---|---|---:|']
    for r in historical:
        lines.append(f"| {r['protocol']} | {r['method']} | {r['mean']:.4f} |")
    lines += ['', 'Historical models are locally reproduced with compatible source splits; they',
              'are stronger nonlinear comparators, not capacity-matched Ridge readouts.', '',
              '## Cost and integrity', '',
              f"The frozen teacher has **{features['teacher_parameters']:,} parameters**, with original",
              f"weights **{features['teacher_weight_bytes']:,} bytes**. Combined readout adds only 2,415",
              'coefficients/intercepts, but the teacher is required at inference for this screen.',
              'This exceeds the target model budget and must not be called a compact model.',
              'The 36-feature color-only control does not require the teacher.', '',
              '1,230 vectors of 768 values were extracted from existing 128px source RGB,',
              'upsampled to 224px. This adds no image detail. CLS and mean patch tokens are',
              'concatenated; no teacher fine-tuning. All vectors replay exactly at the same',
              'device and batch size. Original teacher code/weights hashes and notices checked.',
              'Pretraining overlap with the benchmark remains unknown.', '',
              f"Feature extraction took {features['seconds']:.3f}s in this run, peak allocated CUDA",
              f"memory {features['peak_extraction_allocated_mib']:.2f} MiB. This is batch preprocessing,",
              'not batch-1 or end-to-end product latency. No export or deployment optimization.', '',
              f"Audit: {audit['exact_prediction_arrays']} exact prediction arrays;",
              f"{audit['independent_scalar_delta_e00_cases']} independent scalar color cases;",
              f"{audit['independent_coverage_rows']} independent fixed-coverage rows; all 108 weighted",
              'normal equations checked and independently solved. Fit-only scalers, weights,',
              'shuffles and same-camera alpha selection verified.', '',
              'Five-neighbor input distance is uncalibrated density, not expected DeltaE00.',
              '[All candidate risk curves](all_candidate_risk_coverage.csv) retain every alpha.',
              'No ordinary facial-phone or new independent skin-accuracy conclusion follows.', '',
              '[Protocol](../../research/skin_teacher_readout_protocol_v1.md), [audit](audit.json),',
              '[next decision](../../research/skin_teacher_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    print(json.dumps({'selected_readouts': len(selected), 'candidate_fits': audit['fits'], 'curve_points': 19008}))


if __name__ == '__main__':
    main()
