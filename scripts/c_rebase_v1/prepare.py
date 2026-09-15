"""Create C_REBASE_V1 inputs once, before any training; never overwrite locks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'docs/benchmarks/mskcc_error_floor_audit_2026_09_15'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--private-rows', type=Path, required=True)
    parser.add_argument('--private-output', type=Path, required=True)
    args = parser.parse_args()
    out = ROOT / 'experiments/c_rebase_v1'
    rows = json.loads(args.private_rows.read_text())
    assert rows == sorted(rows, key=lambda r: r['image']) and len(rows) == 966
    with np.load(AUDIT / 'data/capture_features_anonymized.npz') as z:
        data = {k: z[k] for k in z.files}
    assert np.array_equal(data['row_index'], np.arange(966))
    assert np.array_equal(data['target'], np.array([r['target'] for r in rows]))
    ids = sorted({r['patient'] for r in rows})
    assert len(ids) == 24
    alias = {p: f'P{i+1:02d}' for i, p in enumerate(ids)}
    assert np.array_equal(data['patient'], [alias[r['patient']] for r in rows])
    devices = {p: {r['device'] for r in rows if r['patient'] == p} for p in ids}
    assert all(len(v) == 1 for v in devices.values())
    devices = {p: next(iter(v)) for p, v in devices.items()}
    salt = 'LUMA_C_REBASE_V1|subject|'
    order = lambda p: hashlib.sha256((salt + p).encode()).hexdigest()
    slr = sorted([p for p in ids if devices[p] == 'SLR'], key=order)
    ipod = sorted([p for p in ids if devices[p] == 'ipod'], key=order)
    assert (len(slr), len(ipod)) == (8, 16)
    assignment = {p: i % 6 for i, p in enumerate(slr)}
    for p in ipod:
        f = next(f for f in range(6) if list(assignment.values()).count(f) < 4)
        assignment[p] = f
    people = [{'patient': alias[p], 'fold': assignment[p], 'device': devices[p],
               'images': sum(r['patient'] == p for r in rows)} for p in ids]
    mapping = {'experiment': 'C_REBASE_V1', 'immutable': True,
               'algorithm': 'Within device sort SHA256(salt + original patient ID); SLR round robin i%6; iPod fill lowest-index fold with capacity <4. No target or OOF balancing.',
               'salt': salt, 'people': people, 'row_index': list(range(966)),
               'fold_by_row': [assignment[r['patient']] for r in rows]}
    mapping['fold_counts'] = [{
        'fold': f, 'people': 4,
        'images': sum(assignment[r['patient']] == f for r in rows),
        'devices': dict(Counter(r['device'] for r in rows if assignment[r['patient']] == f)),
        'image_types': dict(Counter(r['image_type'] for r in rows if assignment[r['patient']] == f)),
    } for f in range(6)]
    private = {**mapping, 'original_people': [{'original_patient_id': p, **people[i]} for i, p in enumerate(ids)],
               'private_rows_sha256': sha(args.private_rows)}
    write(args.private_output, private)
    mapping['private_full_mapping_sha256'] = sha(args.private_output)
    write(out / 'person_folds.lock.json', mapping)
    config = {
        'experiment': 'C_REBASE_V1', 'status': 'FIXED_BEFORE_ANY_OOF_RESULTS',
        'historical_C_status': 'HISTORICAL_NOT_REPRODUCIBLE',
        'seed': 17, 'architecture': [36, 64, 3], 'activation': 'ReLU',
        'initialization': 'torch_nn_Linear_default', 'epochs': 100, 'phase_boundary': 50,
        'optimizer_boundary': 'continue_state_and_rng', 'batch_size': 64,
        'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .0001,
                      'betas': [.9, .999], 'eps': 1e-8, 'amsgrad': False,
                      'foreach': False, 'fused': False},
        'gradient_clip_norm': 5., 'scheduler': None, 'device': 'cpu', 'num_threads': 1,
        'dtype': 'float32', 'loss': 'mean_squared_error_standardized_Lab',
        'normalization': {'method': 'training_rows_population_mean_std_float64', 'std_floor': 1e-6},
        'shuffle': 'numpy_RandomState_seed_plus_fold_permutation_each_epoch',
        'sampling': 'uniform_images_no_weights', 'early_stopping': False,
        'versions': {'python': platform.python_version(), 'pytorch': torch.__version__,
                     'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'preprocessing': {
            'source': 'scripts/error_floor/capture_audit.py:decode_one,color36',
            'decode': 'Pillow EXIF transpose; embedded ICC to sRGB; unprofiled RGB assumed sRGB',
            'roi': 'central square side=floor(0.8*min(width,height)); floor centered; no facial parser on clinical/dermoscopic inputs',
            'resize': '128x128 Pillow LANCZOS', 'input': 'encoded sRGB channels /255, float64',
            'color36': '27 quantile-major RGB values q=.01,.05,.1,.25,.5,.75,.9,.95,.99; RGB mean; population std; RG,RB,GB correlations (constant channels=0)',
            'augmentations': None,
        },
        'target': {'space': 'CIELAB', 'illuminant': 'D65', 'observer': '10 degree',
                   'definition': 'arithmetic coordinate mean of the three S4 instrument assessments for exact site/tag; shared across site images'},
        'observed_color': {'space': 'sRGB-derived CIELAB', 'illuminant': 'D65', 'observer': '2 degree',
                           'cross_observer_delta_e_allowed': False},
        'training_scope': {'images': 966, 'people': 24, 'sites': 248, 'roles': ['TRAIN']},
        'selection': 'None: one fixed recipe; no OOF-based changes within this experiment',
        'reproduction': {'fold': 0, 'fresh_process': True, 'expected': 'bitwise identical predictions and tensor state on frozen CPU runtime', 'fallback_atol': 1e-6, 'fallback_rtol': 0},
    }
    write(out / 'config.lock.json', config)
    capture = json.loads((AUDIT / 'data/capture.json').read_text())
    vals = [r['original_central_mean_delta_e00']['mean'] for r in capture['per_site'] if r['pairs'] > 0]
    assert len(vals) == 247
    lock = {'experiment': 'C_REBASE_V1', 'phase': 'BEFORE_ANY_OOF_RESULTS',
            'capture_quartiles': {'feature': 'within_site_observed_mean_pair_mean_delta_e00',
                'edges': np.quantile(vals, [.25, .5, .75]).tolist(), 'unique_sites': 247,
                'assignment': 'Q1 <=q25; Q2 (q25,q50]; Q3 (q50,q75]; Q4 >q75; singleton unassigned'},
            'prior_counterfactual_thresholds': {
                'site_reference_pair_mean_p80': 3.562761642774102,
                'site_capture_pair_median_color_mean_p80': 15.084910778391388,
                'image_original_high_plus_low_clip_p80': .004319775552172747},
            'bootstrap': {'seed': 9123, 'draws': 2000}, 'frozen_before_oof': True,
            'interpretation': 'diagnostic only; no deployment thresholds or causal contribution claims',
            'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [
                AUDIT / 'data/capture.json', AUDIT / 'data/instrument.json',
                AUDIT / 'data/diagnostic_subset_masks.anonymized.npz']}}
    write(out / 'diagnostic_plan.lock.json', lock)
    source = ROOT / 'data/public/mskcc_skin_v1'
    raw_sha = {name: sha(source / name) for name in ['s1.csv', 's2.csv', 's4.csv', 's7.csv', 'mskcc-skin-tone-labeling-dataset.csv']}
    manifest = [{'row_index': i, 'image': str(data['image'][i]), 'patient': str(data['patient'][i]),
                 'site': str(data['site'][i]), 'fold': assignment[r['patient']]} for i, r in enumerate(rows)]
    write(out / 'train_manifest.lock.json', {'experiment': 'C_REBASE_V1', 'rows': manifest,
          'source_sha256': raw_sha, 'private_rows_sha256': sha(args.private_rows),
          'original_role_receipt_sha256': sha(AUDIT / 'source_provenance/split_receipt.json')})
    print(json.dumps({'fold_counts': mapping['fold_counts'], 'quartile_edges': lock['capture_quartiles']['edges'],
                      'mapping_sha256': sha(out / 'person_folds.lock.json')}, indent=2))


if __name__ == '__main__':
    main()
