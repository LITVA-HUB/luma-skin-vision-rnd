"""Compare a preserved original fold and independently trained fresh fold."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--repeat', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    first = json.loads((args.original / 'fold_result.json').read_text())
    second = json.loads((args.repeat / 'fold_result.json').read_text())
    assert first['hashes'] == second['hashes']
    arrays = {}
    with np.load(args.original / 'predictions.npz') as a, np.load(args.repeat / 'predictions.npz') as b:
        assert a.files == b.files
        for key in a.files:
            arrays[key] = bool(np.array_equal(a[key], b[key]))
        maximum = float(np.max(np.abs(a['predicted_lab'] - b['predicted_lab'])))
    state = {}
    normalization = {}
    for epoch in [50, 100]:
        a = torch.load(args.original / f'epoch_{epoch:03d}.pt', weights_only=True, map_location='cpu')
        b = torch.load(args.repeat / f'epoch_{epoch:03d}.pt', weights_only=True, map_location='cpu')
        state[str(epoch)] = {k: torch.equal(v, b['model_state_dict'][k]) for k, v in a['model_state_dict'].items()}
        normalization[str(epoch)] = {k: torch.equal(v, b['normalization'][k]) for k, v in a['normalization'].items()}
        assert a['numpy_shuffle_state'] == b['numpy_shuffle_state']
        assert torch.equal(a['torch_rng_state'], b['torch_rng_state'])
    equal = all(arrays.values()) and all(all(x.values()) for x in state.values())
    assert equal and all(all(x.values()) for x in normalization.values())
    assert first['metrics'] == second['metrics']
    report = {'experiment': 'C_REBASE_V1', 'status': 'BITWISE_REPRODUCED_FOLD_0',
              'independent_fresh_process': True, 'fold': 0, 'images': first['holdout_images'],
              'people': len(first['holdout_people']), 'configuration_hashes': first['hashes'],
              'all_prediction_arrays_bitwise_equal': arrays,
              'maximum_absolute_prediction_difference_Lab': maximum,
              'metric_values_exactly_equal': True, 'model_tensors_equal': state,
              'normalization_tensors_equal': normalization,
              'final_model_state_sha256': first['final_state_sha256'],
              'original_prediction_file_sha256': sha(args.original / 'predictions.npz'),
              'repeat_prediction_file_sha256': sha(args.repeat / 'predictions.npz'),
              'predeclared_fallback_atol': 1e-6, 'fallback_used': False,
              'optimizer_trajectory_logs_equal': json.loads((args.original / 'training_log.json').read_text()) == json.loads((args.repeat / 'training_log.json').read_text()),
              'limitations': 'Exact within this frozen CPU runtime. No assertion of cross-platform bitwise equality.',
              'verifier_sha256': sha(Path(__file__))}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'maximum_absolute_prediction_difference_Lab': maximum,
                      'metrics': first['metrics']}, indent=2))


if __name__ == '__main__':
    main()
