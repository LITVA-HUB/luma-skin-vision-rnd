"""Post-hoc source gradient diagnostic; no fitting or parameter mutation."""
import json
from pathlib import Path
import numpy as np
import torch
from skin_mskcc_data import ROOT, sha
from skin_mskcc_train_pixels import digest
from skin_local_teacher_features import load, pack, OUT
from skin_local_teacher_model import LocalTeacherColor
from skin_capture_model import MODES
from skin_pair_invariance import pair_indices


def gradient_relation(color, auxiliary):
    cc = aa = ca = 0.
    for c, a in zip(color, auxiliary, strict=True):
        if c is not None: cc += float(c.double().square().sum())
        if a is not None: aa += float(a.double().square().sum())
        if c is not None and a is not None: ca += float((c.double()*a.double()).sum())
    cn, an = cc**.5, aa**.5
    return {'color_norm': cn, 'weighted_aux_norm': an, 'gradient_dot': ca,
            'cosine': ca/(cn*an) if cn and an else None,
            'weighted_aux_to_color_norm_ratio': an/cn if cn else None,
            'color_derivative_along_negative_aux_gradient': -ca}


def main():
    protocol = ROOT/'docs/research/skin_teacher_conflict_protocol_v1.md'
    out = OUT/'gradient_diagnostic'; out.mkdir(exist_ok=False)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    data = load('train'); a, b = pair_indices(data['site'], 64, np.random.default_rng(81017))
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), protocol,
        ROOT/'tests/test_skin_teacher_conflict.py', ROOT/'scripts/skin_local_teacher_model.py',
        ROOT/'scripts/skin_local_teacher_features.py', ROOT/'scripts/skin_capture_model.py',
        ROOT/'scripts/skin_pair_invariance.py', OUT/'source_lock.json']}
    records = sorted(OUT.glob('mixed__*/result.json')); assert len(records) == 12
    results = []
    for path in records:
        r = json.loads(path.read_bytes()); bindings[str(path.relative_to(ROOT))] = sha(path)
        checkpoint = ROOT/'experiments/runs/skin_local_teacher_v1'/path.parent.name/'best.pt'
        assert sha(checkpoint) == r['best_sha256']
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        model = LocalTeacherColor(r['arch']).eval(); model.load_state_dict(saved['state'])
        before = digest(model.state_dict()); named = list(model.named_parameters()); params = [p for _, p in named]
        groups = {name: [i for i, (key, _) in enumerate(named) if key.startswith(name+'.')] for name in ['local', 'adapter', 'context', 'votes', 'gate']}
        groups['all'] = list(range(len(named)))
        x = torch.from_numpy(pack(data, r['arch']))
        y = (torch.from_numpy(data['target']).float()-saved['target_mean'])/saved['target_std']
        mode = torch.tensor([MODES.index(str(v)) for v in data['mode']])
        for j, offset in enumerate(range(0, 64, 16)):
            ix = torch.from_numpy(np.r_[a[offset:offset+16], b[offset:offset+16]])
            prediction, logits, _ = model(x[ix]); color = (prediction-y[ix]).square().mean()
            auxiliary = .1*torch.nn.functional.cross_entropy(logits, mode[ix])
            cg = torch.autograd.grad(color, params, retain_graph=True, allow_unused=True)
            ag = torch.autograd.grad(auxiliary, params, allow_unused=True)
            for group, indices in groups.items():
                results.append({'arm': r['arch'], 'seed': r['seed'], 'batch': j, 'group': group,
                    'color_mse': float(color.detach()), 'weighted_mode_ce': float(auxiliary.detach()),
                    **gradient_relation([cg[i] for i in indices], [ag[i] for i in indices])})
        assert digest(model.state_dict()) == before and all(p.grad is None for p in params)
        del model, x, y, cg, ag
    summary = []
    for arm in ['plain', 'aligned', 'global', 'shuffled']:
        rr = [r for r in results if r['arm'] == arm and r['group'] == 'all']
        summary.append({'arm': arm, 'batches_across_three_models': len(rr),
            'negative_dot_fraction': float(np.mean([r['gradient_dot'] < 0 for r in rr])),
            'mean_cosine': float(np.mean([r['cosine'] for r in rr])),
            'mean_aux_to_color_norm_ratio': float(np.mean([r['weighted_aux_to_color_norm_ratio'] for r in rr]))})
    write = lambda p, r: p.write_text(json.dumps(r, indent=2)+'\n', encoding='utf8')
    write(out/'results.json', {'scope': 'POST-HOC TRAIN gradients; no optimization or causal accuracy inference',
        'bindings': bindings, 'records': results, 'summary': summary,
        'unchanged_model_states': len(records), 'reserved_or_validation_inputs_used': False})
    for row in summary: print(json.dumps(row))


if __name__ == '__main__': main()
