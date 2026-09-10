"""Equal-query fixed versus adaptive correction search on development data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def fixed_select(model, cache, steps):
    if steps not in (1, 2, 4):
        raise ValueError('Expected1,2,4 stages')
    point = cache['point_action']
    proposals = []
    for stage, radius in enumerate((.24, .06, .03, .015)[:steps]):
        proposals.append((point[:, None]+radius*model.search_offsets[None]).clamp(-2, 2))
        if stage:
            proposals.append(point[:, None])
    proposals = torch.cat(proposals, 1)
    risks = model.query(cache, proposals)['angular_risk']
    index = risks.argmin(-1)
    rows = torch.arange(len(point), device=point.device)
    action = proposals[rows, index]
    action = torch.where(cache['valid'][:, None], action, torch.zeros_like(action))
    return {'action': action, 'risk': risks[rows, index], 'valid': cache['valid'],
            'query_count': proposals.shape[1]}


def main(args):
    from cc_v2_statistics import read_npz_rows
    from cc_v4_experiment import action_rgb
    from cc_v5_model import CorrectionEvidenceNet
    from cc_v5_report import ARMS, reproduction_degrees, selective, summarize

    from luma_skin_vision.cc.core import angular
    from luma_skin_vision.data import sha256
    from luma_skin_vision.experiment import write_json

    root = Path(__file__).resolve().parents[1]
    protocol = root/'docs/research/cc_v5_search_control_protocol.md'
    out = Path(args.out)
    if out.exists():
        raise ValueError('Use new output directory; no result overwrites')
    torch.set_num_threads(2)
    data = root/'data/processed/cc128'
    rows = json.loads((data/'cube_manifest.json').read_text(encoding='utf-8'))
    ix = np.array([i for i, r in enumerate(rows) if r['subset'] == 'val'])
    if len(ix) != 119:
        raise ValueError('Unexpected validation population')
    x = gt = None
    result = {'status': 'REUSED DEVELOPMENT VALIDATION; NO NEW CAMERA GT',
              'protocol_sha256': sha256(protocol), 'script_sha256': sha256(__file__),
              'device': 'CPU', 'torch': torch.__version__, 'bindings': [], 'records': []}
    payloads = []
    for run_name in args.runs:
        run = Path(run_name)
        config = json.loads((run/'config.json').read_text(encoding='utf-8'))
        if config['validation_ids'] != [rows[i]['id'] for i in ix]:
            raise ValueError('Validation identity/order mismatch')
        for name, expected in config['data_sha256'].items():
            if sha256(data/name) != expected:
                raise ValueError('Source data changed')
        for name, expected in config['scripts_sha256'].items():
            if sha256(root/'scripts'/name) != expected:
                raise ValueError('Frozen executable changed')
        manifest = json.loads((run/'artifact_manifest.json').read_text(encoding='utf-8'))['sha256']
        for name, expected in manifest.items():
            if sha256(run/name) != expected:
                raise ValueError('Frozen run changed: '+name)
        if x is None:
            x = torch.from_numpy(read_npz_rows(data/'cube.npz', 'images', ix, 2234).astype(np.float32))
            gt = read_npz_rows(data/'cube.npz', 'gt', ix, 2234).astype(np.float64)
        seed = config['arguments']['seed']
        for arm in ARMS[1:]:
            path = run/arm/'best.pt'
            result['bindings'].append({'seed': seed, 'arm': arm, 'checkpoint_sha256': sha256(path),
                                       'data_sha256': config['data_sha256'], 'scripts_sha256': config['scripts_sha256']})
            model = CorrectionEvidenceNet(mode=arm.split('_')[0]).eval()
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            arrays = {(policy, steps): {'action': [], 'risk': [], 'valid': []}
                      for policy in ('adaptive', 'fixed') for steps in (1, 2, 4)}
            with torch.no_grad():
                for start in range(0, len(x), 16):
                    cache = model.encode(x[start:start+16])
                    for steps, budget in ((1, 25), (2, 51), (4, 103)):
                        selections = {'adaptive': model.select(cache, steps), 'fixed': fixed_select(model, cache, steps)}
                        if steps == 1 and not torch.allclose(selections['adaptive']['action'], selections['fixed']['action'], atol=1e-6, rtol=0):
                            raise ValueError('One-stage control mismatch')
                        for policy, selected in selections.items():
                            if selected['query_count'] != budget:
                                raise ValueError('Unequal candidate budget')
                            for key in ('action', 'risk', 'valid'):
                                arrays[(policy, steps)][key].append(selected[key].cpu().numpy())
            for (policy, steps), values in arrays.items():
                values = {k: np.concatenate(v) for k, v in values.items()}
                if not values['valid'].all():
                    raise ValueError('Unexpected rejected development row')
                pred = action_rgb(values['action'].astype(np.float64))
                error = reproduction_degrees(pred, gt)
                record = {'seed': seed, 'arm': arm, 'policy': policy, 'steps': steps,
                          'queries': {1: 25, 2: 51, 4: 103}[steps],
                          'reproduction': summarize(error), 'recovery': summarize(angular(pred, gt)),
                          'selective': selective(error, values['risk'])}
                result['records'].append(record)
                payloads.append((f's{seed}_{arm}_{policy}_{steps}.npz', {**values, 'pred': pred, 'reproduction': error}))
            print(json.dumps({'seed': seed, 'arm': arm, 'completed_equal_query_comparison': True}), flush=True)
    out.mkdir(parents=True)
    for filename, payload in payloads:
        np.savez_compressed(out/filename, **payload)
    write_json(out/'summary.json', result)
    write_json(out/'manifest.json', {'sha256': {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    main(parser.parse_args())
