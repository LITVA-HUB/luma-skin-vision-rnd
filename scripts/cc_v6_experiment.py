"""Paired canonical-frame correction evidence development experiment.

V5 executable bytes remain unchanged. See docs/research/cc_v6_combination_protocol.md.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v3_experiment import DATA_HASHES, source_rows
from cc_v4_experiment import field_weight
from cc_v5_model import analytic_costs
from cc_v6_evaluate import evaluate
from cc_v6_model import CanonicalEvidenceNet

from luma_skin_vision.cc.v2_experiment import source_snapshot
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('point', 'posterior_random', 'action_random', 'transport_random')


def capture_state(model, optimizer, scheduler):
    return copy.deepcopy({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()})


def restore_state(state, model, optimizer, scheduler):
    value = copy.deepcopy(state)
    model.load_state_dict(value['model'])
    optimizer.load_state_dict(value['optimizer'])
    scheduler.load_state_dict(value['scheduler'])


def state_digest(state):
    digest = hashlib.sha256()

    def visit(value):
        digest.update(type(value).__name__.encode())
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            digest.update(str((cpu.dtype, tuple(cpu.shape))).encode())
            digest.update(cpu.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(state)
    return digest.hexdigest()


def draw_actions(point, center, generator):
    options = {'device': point.device, 'dtype': point.dtype, 'generator': generator}
    local = center.detach()[:, None] + (torch.rand(len(point), 16, 2, **options) * 2 - 1) * .4
    global_ = (torch.rand(len(point), 16, 2, **options) * 2 - 1) * 1.5
    return torch.cat((point.detach()[:, None], local, global_), 1).clamp(-2, 2)


def physical_targets(loggt, actions, with_gradient=False):
    target = analytic_costs(loggt[:, None], actions)
    derivative = None
    if with_gradient:
        derivative = torch.autograd.grad(target['sin2'].sum(), actions)[0].detach()
    return {key: val.squeeze(-1).detach() for key, val in target.items()}, derivative


def train_epoch(model, optimizer, scheduler, x, gt, train_ix, epoch, seed, batch, arm, warmup):
    model.train()
    device = x.device
    order_rng = torch.Generator(device=device).manual_seed(seed * 100000 + epoch * 10)
    aug_rng = torch.Generator(device=device).manual_seed(seed * 100000 + epoch * 10 + 1)
    action_rng = torch.Generator(device=device).manual_seed(seed * 100000 + epoch * 10 + 2)
    order = train_ix[torch.randperm(len(train_ix), device=device, generator=order_rng)]
    total = np.zeros(4)
    seen = 0
    for ix in order.split(batch):
        flip = torch.rand(len(ix), 1, 1, 1, device=device, generator=aug_rng) < .5
        exposure = (torch.rand(len(ix), 1, 1, 1, device=device, generator=aug_rng) - .5).exp()
        xb = torch.where(flip, x[ix].flip(-1), x[ix]) * exposure
        loggt = gt[ix].log()
        loggt = loggt[:, [0, 2]] - loggt[:, 1:2]
        optimizer.zero_grad(set_to_none=True)
        cache = model.encode(xb)
        loggt = loggt - cache['anchor_logchroma']
        point = analytic_costs(loggt[:, None], cache['point_action'][:, None])['angular'].mean()
        ang, cost, gradient = (point.new_zeros(()) for _ in range(3))
        weight = field_weight(epoch) if warmup == 20 else min(1., max(0., (epoch - warmup) / 20))
        if arm != 'point' and weight:
            with torch.no_grad():
                selected = model.select(cache, steps=2)['action'].detach()
            center = selected if arm in {'transport_policy', 'transport_gradient'} else cache['point_action']
            actions = draw_actions(cache['point_action'], center, action_rng)
            with_gradient = arm == 'transport_gradient'
            actions.requires_grad_(with_gradient)
            field = model.query(cache, actions)
            target, desired_gradient = physical_targets(loggt, actions, with_gradient)
            ang = (field['angular_risk'] - target['angular']).square().mean()
            cost = (field['sin2_risk'] - target['sin2']).square().mean()
            if with_gradient:
                actual = torch.autograd.grad(field['sin2_risk'].sum(), actions, create_graph=True)[0]
                gradient = (actual - desired_gradient).square().mean()
        loss = point + weight * (.05 * ang + 25 * cost + 25 * gradient)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss {arm} epoch{epoch}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        total += np.array([v.item() for v in (point, ang, cost, gradient)]) * len(ix)
        seen += len(ix)
    scheduler.step()
    return dict(zip(('point', 'angular_mse', 'sin2_mse', 'gradient_mse'), (total / seen).tolist()))


def run(args):
    if args.epochs <= args.warmup or args.warmup < 1 or args.batch < 2:
        raise ValueError('Require epochs > warmup >=1 and batch>=2')
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('Immutable run directory already exists')
    data = Path(args.data).resolve()
    if {name: sha256(data / name) for name in DATA_HASHES} != DATA_HASHES:
        raise ValueError('Frozen source input hashes changed')
    rows = json.loads((data / 'cube_manifest.json').read_text(encoding='utf-8'))
    selected, train_ix, val_ix = source_rows(rows)
    if (len(rows), len(train_ix), len(val_ix)) != (2234, 1126, 119):
        raise ValueError('Source role counts changed')
    out.mkdir(parents=True)
    identity, snapshot = source_snapshot(out)
    dependencies = ['cc_v6_experiment.py', 'cc_v6_model.py', 'cc_v6_evaluate.py', 'cc_v5_experiment.py', 'cc_v5_model.py', 'cc_v4_experiment.py', 'cc_v4_model.py',
                    'cc_v3_experiment.py', 'cc_v3_model.py', 'cc_v2_statistics.py']
    hashes = {}
    for name in dependencies:
        dest = out / 'source_snapshot/scripts' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'scripts' / name, dest)
        hashes[name] = sha256(dest)
    shutil.copyfile(ROOT / 'docs/research/cc_v6_combination_protocol.md', out / 'protocol.md')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    x = torch.from_numpy(read_npz_rows(data / 'cube.npz', 'images', selected, 2234).astype(np.float32)).to(device)
    gt = torch.from_numpy(read_npz_rows(data / 'cube.npz', 'gt', selected, 2234).astype(np.float32)).to(device)
    if not torch.isfinite(x).all() or (x < 0).any() or not torch.isfinite(gt).all() or (gt <= 0).any():
        raise ValueError('Nonfinite or invalid source data')
    train_ix, val_ix = torch.as_tensor(train_ix, device=device), torch.as_tensor(val_ix, device=device)
    config = {'arguments': vars(args), 'arms': ARMS, 'source_identity': identity, 'snapshot': snapshot,
              'scripts_sha256': hashes, 'data_sha256': DATA_HASHES, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
              'source_cache_mib': (x.numel() * x.element_size() + gt.numel() * gt.element_size()) / 2**20,
              'deterministic_algorithms': True, 'cublas_workspace': os.environ['CUBLAS_WORKSPACE_CONFIG'],
              'primary': (args.epochs, args.warmup, args.batch) == (120, 20, 32),
              'train_ids': [rows[selected[int(i)]]['id'] for i in train_ix.cpu()],
              'validation_ids': [rows[selected[int(i)]]['id'] for i in val_ix.cpu()],
              'limitations': 'Reused source validation; no independent test/calibration/camera/skin evidence'}
    write_json(out / 'config.json', config)
    model = CanonicalEvidenceNet(mode='posterior', frame=args.frame).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00002)
    initial = capture_state(model, optimizer, scheduler)
    first_training = train_epoch(model, optimizer, scheduler, x, gt, train_ix, 1, args.seed, args.batch, 'point', args.warmup)
    first = capture_state(model, optimizer, scheduler)
    restore_state(initial, model, optimizer, scheduler)
    replay_training = train_epoch(model, optimizer, scheduler, x, gt, train_ix, 1, args.seed, args.batch, 'point', args.warmup)
    replay = capture_state(model, optimizer, scheduler)
    receipt = {'first_sha256': state_digest(first), 'replay_sha256': state_digest(replay),
               'loss_equal': first_training == replay_training}
    receipt['bitwise_equal'] = receipt['first_sha256'] == receipt['replay_sha256'] and receipt['loss_equal']
    write_json(out / 'warmup_replay.json', receipt)
    if not receipt['bitwise_equal']:
        raise RuntimeError('Exact common-state warmup replay failed')
    del initial, first, replay
    for epoch in range(2, args.warmup + 1):
        stats = train_epoch(model, optimizer, scheduler, x, gt, train_ix, epoch, args.seed, args.batch, 'point', args.warmup)
        if epoch % 5 == 0 or epoch == args.warmup:
            print(json.dumps({'phase': 'warmup', 'epoch': epoch, 'training': stats}), flush=True)
    common = capture_state(model, optimizer, scheduler)
    common_hash = state_digest(common)
    torch.save(common, out / 'common_start.pt')
    write_json(out / 'common_start.json', {'state_sha256': common_hash, 'epoch': args.warmup,
                                         'file_sha256': sha256(out / 'common_start.pt')})
    results = {}
    for arm in ARMS:
        folder = out / arm
        folder.mkdir()
        restore_state(common, model, optimizer, scheduler)
        if state_digest(capture_state(model, optimizer, scheduler)) != common_hash:
            raise RuntimeError('Arm did not restore the complete common state')
        model.mode = 'posterior' if arm == 'point' else arm.split('_')[0]
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        best, best_epoch = math.inf, None
        for epoch in range(args.warmup, args.epochs + 1):
            stats = None
            if epoch > args.warmup:
                stats = train_epoch(model, optimizer, scheduler, x, gt, train_ix, epoch, args.seed, args.batch, arm, args.warmup)
            metrics, arrays = evaluate(model, x[val_ix], gt[val_ix], args.batch)
            key = metrics['base_reproduction']['mean'] if arm == 'point' else metrics['stages']['2']['reproduction']['mean']
            if metrics['valid_fraction'] < .99:
                raise RuntimeError('Unexpected invalid source rows')
            if key < best:
                best, best_epoch = key, epoch
                torch.save(model.state_dict(), folder / 'best.pt')
                np.savez_compressed(folder / 'best_validation.npz', **arrays)
                write_json(folder / 'best_metrics.json', {'epoch': epoch, 'arm': arm, **metrics})
            with (folder / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'epoch': epoch, 'training': stats, 'validation': metrics}) + '\n')
            if epoch % 10 == 0 or epoch == args.epochs:
                print(json.dumps({'arm': arm, 'epoch': epoch, 'val_repro': key,
                                  'best': best, 'seconds': round(time.perf_counter() - started, 1)}), flush=True)
        torch.save(model.state_dict(), folder / 'last.pt')
        np.savez_compressed(folder / 'last_validation.npz', **arrays)
        write_json(folder / 'last_metrics.json', {'epoch': args.epochs, 'arm': arm, **metrics})
        results[arm] = {'best_mean_reproduction': best, 'best_epoch': best_epoch, 'last_mean_reproduction': key,
                        'common_start_state_sha256': common_hash,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'elapsed_seconds': time.perf_counter() - started,
                        'timing_scope': 'Training plus epoch validation and checkpoint I/O; not deployment latency',
                        'training_peak_mib_including_cache_and_common_state': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
                        'best_checkpoint_sha256': sha256(folder / 'best.pt'),
                        'best_checkpoint_bytes': (folder / 'best.pt').stat().st_size}
        write_json(folder / 'result.json', results[arm])
        if state_digest(common) != common_hash:
            raise RuntimeError('Training mutated the shared starting state')
    write_json(out / 'result.json', {'status': 'complete', 'arms': results})
    write_json(out / 'artifact_manifest.json', {'sha256': {
        str(p.relative_to(out)).replace('\\', '/'): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({'status': 'complete', 'arms': results}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--data', default='data/processed/cc128')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--frame', choices=('none', 'sog'), default='sog')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    arguments = parser.parse_args()
    existed = Path(arguments.out).exists()
    try:
        run(arguments)
    except Exception as exc:
        folder = Path(arguments.out)
        if not existed and folder.is_dir() and not (folder / 'result.json').exists():
            write_json(folder / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
