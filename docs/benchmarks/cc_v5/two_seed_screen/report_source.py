"""Independently rescore completed V5 runs; no training or image decoding."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ARMS = ('point', 'posterior_random', 'action_random', 'transport_random',
        'transport_policy', 'transport_gradient')


def reproduction_degrees(pred, gt):
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 3 or pred.shape != gt.shape:
        raise ValueError('Aligned Nx3 RGB required')
    if not np.isfinite([pred, gt]).all() or (pred <= 0).any() or (gt <= 0).any():
        raise ValueError('Positive finite RGB required')
    ratio = gt / pred
    ratio /= ratio.max(-1, keepdims=True)
    r, g, b = ratio.T
    return np.rad2deg(np.arctan2(np.sqrt((r-g)**2 + (g-b)**2 + (b-r)**2), r+g+b))


def summarize(error):
    error = np.asarray(error, dtype=np.float64)
    q25, median, q75 = np.percentile(error, [25, 50, 75])
    quarter = max(1, int(len(error) / 4))
    ordered = np.sort(error)
    return {'mean': float(error.mean()), 'median': float(median),
            'trimean': float((q25 + 2*median + q75) / 4),
            'best25': float(ordered[:quarter].mean()), 'worst25': float(ordered[-quarter:].mean()),
            'p90': float(np.percentile(error, 90)), 'p95': float(np.percentile(error, 95)),
            'over10': int((error > 10).sum()), 'count': len(error)}


def selective(error, risk):
    error, risk = np.asarray(error), np.asarray(risk)
    if error.ndim != 1 or not len(error) or error.shape != risk.shape or not np.isfinite([error, risk]).all():
        raise ValueError('Aligned nonempty finite errors and risk required')
    order = np.argsort(risk, kind='stable')
    ordered = error[order]
    fixed = {}
    for coverage in (100, 95, 90, 80, 70, 60):
        k = max(1, int(len(error) * coverage / 100))
        fixed[str(coverage)] = {'accepted': k, 'coverage': k/len(error), **summarize(ordered[:k])}
    return {'fixed': fixed, 'curve': (np.cumsum(ordered)/np.arange(1, len(error)+1)).tolist()}


def verify_manifest(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'artifact_manifest.json').read_text(encoding='utf-8'))['sha256']
    for name, expected in manifest.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Invalid or changed artifact: {name}')
    return manifest


def generate(args):
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError('Report output is immutable; choose a new directory')
    data = Path(args.data)
    if {name: sha256(data/name) for name in DATA_HASHES} != DATA_HASHES:
        raise ValueError('Frozen source data changed')
    rows = json.loads((data/'cube_manifest.json').read_text(encoding='utf-8'))
    selected, train, val = source_rows(rows)
    indices = [selected[int(i)] for i in val]
    gt = read_npz_rows(data/'cube.npz', 'gt', indices, 2234).astype(np.float32).astype(np.float64)
    expected_ids = [rows[i]['id'] for i in indices]
    expected_train_ids = [rows[selected[int(i)]]['id'] for i in train]
    results, bindings, archive = [], [], []
    seen_seeds = set()
    script_hashes = None
    for run in map(Path, args.runs):
        manifest = verify_manifest(run)
        config = json.loads((run/'config.json').read_text(encoding='utf-8'))
        result = json.loads((run/'result.json').read_text(encoding='utf-8'))
        common = json.loads((run/'common_start.json').read_text(encoding='utf-8'))
        replay = json.loads((run/'warmup_replay.json').read_text(encoding='utf-8'))
        seed = config['arguments']['seed']
        if not config['primary'] or result['status'] != 'complete' or seed in seen_seeds:
            raise ValueError('Require completed unique-seed primary runs')
        if config['validation_ids'] != expected_ids or config['train_ids'] != expected_train_ids:
            raise ValueError('Role/row ordering mismatch')
        if not replay['bitwise_equal'] or replay['first_sha256'] != replay['replay_sha256']:
            raise ValueError('Warmup did not replay exactly')
        if script_hashes is not None and config['scripts_sha256'] != script_hashes:
            raise ValueError('Seeds ran different executable code')
        script_hashes = config['scripts_sha256']
        seen_seeds.add(seed)
        for name, digest in config['scripts_sha256'].items():
            if manifest.get(f'source_snapshot/scripts/{name}') != digest:
                raise ValueError('Unbound source script')
        bindings.append({'run': str(run), 'seed': seed, 'scripts_sha256': script_hashes,
                         'common_state_sha256': common['state_sha256']})
        for arm in ARMS:
            arm_result = result['arms'][arm]
            if arm_result['common_start_state_sha256'] != common['state_sha256']:
                raise ValueError('Arm common-state mismatch')
            if manifest.get(f'{arm}/best.pt') != arm_result['best_checkpoint_sha256']:
                raise ValueError('Best checkpoint mismatch')
            for checkpoint in ('best', 'last'):
                payload = np.load(run/arm/f'{checkpoint}_validation.npz', allow_pickle=False)
                if payload['valid'].shape != (119,) or not payload['valid'].all():
                    raise ValueError('Invalid validation population')
                pred = payload['base_pred'] if arm == 'point' else payload['pred']
                error = reproduction_degrees(pred, gt)
                stored = payload['base_reproduction'] if arm == 'point' else payload['reproduction']
                discrepancy = float(np.max(np.abs(error-stored)))
                if discrepancy > 1e-6:
                    raise ValueError('Stored reproduction disagrees with independent GT scoring')
                expected_mean = arm_result[f'{checkpoint}_mean_reproduction']
                if abs(float(error.mean()) - expected_mean) > 1e-6:
                    raise ValueError('Result mean disagrees with prediction scoring')
                normpred = pred/np.linalg.norm(pred, axis=-1, keepdims=True)
                normgt = gt/np.linalg.norm(gt, axis=-1, keepdims=True)
                recovery = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(normpred, normgt), axis=-1), (normpred*normgt).sum(-1)))
                record = {'seed': seed, 'arm': arm, 'checkpoint': checkpoint,
                          'reproduction': summarize(error), 'recovery': summarize(recovery),
                          'max_recomputation_difference': discrepancy, 'compute': arm_result,
                          'selective': None if arm == 'point' else selective(error, payload['risk'])}
                stage_errors = []
                for stage in range(4):
                    action = payload['trajectory_actions'][:, stage].astype(np.float64)
                    rgb = np.exp(np.stack((action[:, 0], np.zeros(len(action)), action[:, 1]), -1))
                    stage_errors.append(reproduction_degrees(rgb, gt))
                record['refinement'] = None if arm == 'point' else {
                    'means_1_2_4': [float(stage_errors[i].mean()) for i in (0, 1, 3)],
                    'worsened_1_to_4': int(((stage_errors[3]-stage_errors[0]) > 1e-6).sum())}
                results.append(record)
        for name in manifest:
            if name.endswith('.pt') or name.startswith('source_snapshot/') and not name.startswith('source_snapshot/scripts/'):
                continue
            archive.append((run/name, Path(f'seed{seed}')/name))
        archive.append((run/'artifact_manifest.json', Path(f'seed{seed}')/'original_run_manifest.json'))
    summary = {'status': 'REUSED DEVELOPMENT VALIDATION ONLY', 'seeds': sorted(seen_seeds),
               'source_validation_count': 119, 'bindings': bindings, 'results': results,
               'risk_status': 'Raw uncalibrated model ranking; point control has no trained risk head',
               'limitations': 'No independent test, cross-camera, skin or novelty conclusion; three seeds do not add independent scenes'}
    lines = ['# V5 paired correction-critic source screen', '',
             '**REUSED DEVELOPMENT VALIDATION ONLY.** 1126 real SimpleCube++ training images;119 reused validation images. CC BY4.0. No external weights.', '',
             'All arms share exact warmup model/BN/optimizer/scheduler state per seed. Strict CUDA determinism passed epoch1 replay. Same data and noise streams; gradient loss adds backward compute. No new camera/test labels were consumed.', '',
             '| Seed | Arm | Best epoch | Best mean repro ° | Final mean repro ° | Best raw risk80 ° |',
             '|---|---|---:|---:|---:|---:|']
    for seed in sorted(seen_seeds):
        for arm in ARMS:
            best = next(r for r in results if (r['seed'], r['arm'], r['checkpoint']) == (seed, arm, 'best'))
            last = next(r for r in results if (r['seed'], r['arm'], r['checkpoint']) == (seed, arm, 'last'))
            risk = f"{best['selective']['fixed']['80']['mean']:.4f}" if best['selective'] else 'not trained'
            lines.append(f"| {seed} | {arm} | {best['compute']['best_epoch']} | {best['reproduction']['mean']:.4f} | {last['reproduction']['mean']:.4f} | {risk} |")
    lines += ['', 'Best checkpoints were selected on these same119 images. Both best and final values are retained to expose selection effects. Fixed coverage100/95/90/80/70/60, complete119-point curves, recovery/tail metrics and refinement diagnostics are in summary.json. Nominal80% accepts95/119 images.', '',
              'Point-only risk is omitted because its critic was not trained. No post-hoc risk calibration has been performed. More refinement has no guaranteed improvement; equal-query nonadaptive control remains pending.', '',
              'The model retains3,097,189 parameters. Recorded peak training allocation includes source tensors and the copied common optimizer/model state; elapsed run time includes validation/checkpoint writing. Deployment latency is NOT MEASURED for V5.', '',
              'This screen cannot establish universal camera handling, physical surface-color DeltaE, facial accuracy, published-method superiority or novelty. Independent camera evaluation and conventional calibrated C+ comparison remain required after a stable development result.', '']
    output.mkdir(parents=True)
    write_json(output/'summary.json', summary)
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    for source, relative in archive:
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(__file__, output/'report_source.py')
    write_json(output/'archive_manifest.json', {'sha256': {
        p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'output': str(output), 'seeds': sorted(seen_seeds), 'independent_prediction_records': len(results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data', default='data/processed/cc128')
    generate(parser.parse_args())
