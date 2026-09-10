"""CPU replay of completed V5 checkpoints; verify saved GPU predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v4_experiment import evaluate
from cc_v5_model import CorrectionEvidenceNet
from cc_v5_report import ARMS

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError('Never overwrite a replay receipt')
    torch.set_num_threads(2)
    data = ROOT/'data/processed/cc128'
    rows = json.loads((data/'cube_manifest.json').read_text(encoding='utf-8'))
    indices = np.array([i for i, row in enumerate(rows) if row['subset'] == 'val'])
    if len(indices) != 119:
        raise ValueError('Wrong validation population')
    receipt = {'device': 'CPU', 'torch': torch.__version__, 'script_sha256': sha256(__file__),
               'scope': 'Only119 reused source validation images; no phone/test rows',
               'tolerance': 1e-3, 'models': [], 'status': 'passed'}
    x = gt = None
    for run_arg in args.runs:
        run = Path(run_arg)
        config = json.loads((run/'config.json').read_text(encoding='utf-8'))
        if [rows[i]['id'] for i in indices] != config['validation_ids']:
            raise ValueError('Changed validation order')
        for name, expected in config['data_sha256'].items():
            if sha256(data/name) != expected:
                raise ValueError('Changed source data')
        for name, expected in config['scripts_sha256'].items():
            if sha256(ROOT/'scripts'/name) != expected:
                raise ValueError('Changed executable '+name)
        manifest = json.loads((run/'artifact_manifest.json').read_text(encoding='utf-8'))['sha256']
        for name, expected in manifest.items():
            if sha256(run/name) != expected:
                raise ValueError('Changed run artifact '+name)
        if x is None:
            x = torch.from_numpy(read_npz_rows(data/'cube.npz', 'images', indices, 2234).astype(np.float32))
            gt = torch.from_numpy(read_npz_rows(data/'cube.npz', 'gt', indices, 2234).astype(np.float32))
        for arm in ARMS:
            model = CorrectionEvidenceNet(mode='posterior' if arm == 'point' else arm.split('_')[0]).eval()
            for checkpoint in ('best', 'last'):
                path = run/arm/f'{checkpoint}.pt'
                predictions = run/arm/f'{checkpoint}_validation.npz'
                model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
                _, actual = evaluate(model, x, gt, 16)
                with np.load(predictions, allow_pickle=False) as saved:
                    defects = {key: float(np.max(abs(actual[key].astype(np.float64)-saved[key].astype(np.float64))))
                               for key in ('point_action', 'trajectory_actions', 'trajectory_risk',
                                           'base_reproduction', 'trajectory_reproduction')}
                    same_valid = bool(np.array_equal(actual['valid'], saved['valid']))
                passed = same_valid and all(np.isfinite(v) and v <= 1e-3 for v in defects.values())
                record = {'seed': config['arguments']['seed'], 'arm': arm, 'checkpoint': checkpoint,
                          'checkpoint_sha256': sha256(path), 'predictions_sha256': sha256(predictions),
                          'max_absolute_defects': defects, 'validity_equal': same_valid,
                          'status': 'passed' if passed else 'FAILED'}
                if not passed:
                    receipt['status'] = 'FAILED'
                receipt['models'].append(record)
                print(json.dumps(record), flush=True)
    write_json(out, receipt)
    if receipt['status'] != 'passed':
        raise RuntimeError('One or more CPU replays differ; failure receipt preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
