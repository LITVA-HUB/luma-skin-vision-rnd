"""Seg2 real-data preflights and fixed six-trajectory fine-tuning, queued behind HR/P3."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from chromaseed_head_range_verification import process_alive, require_quiet_host
from skin_face_segment import SkinUNet, augment, skin_loss
from skin_face_transfer_data import (
    SOURCES,
    PairedSampler,
    assemble_batch,
    checked,
    digest,
    load_split,
    read,
)
from skin_face_transfer_study import (
    BATCH,
    BUDGETS,
    COUNTS,
    INITIAL,
    INITIAL_SHA,
    INTERVAL,
    PARAMETERS,
    ROOT,
    RUN,
    WIDTH,
    check_bindings,
    freeze_choices,
    freeze_registration,
    utc,
    verify_registration,
    write_once,
)

PRIOR = [(Path(f'D:/Luma-RnD/{name}/job.json'), ROOT / f'docs/benchmarks/{name}/verification.json')
         for name in ('chromaseed_head_range_v1', 'chromaseed_palette_transfer_v1')]


def check_predecessor_records(records=PRIOR, probe=process_alive):
    for job_path, seal_path in records:
        if not job_path.exists():
            raise RuntimeError(f'Prior worker has not completed: {job_path}')
        job = read(job_path)
        if job.get('status') != 'complete' or probe(job.get('pid')) is not False:
            raise RuntimeError(f'Prior worker must be complete and confirmed dead: {job_path}')
        if not seal_path.exists():
            raise RuntimeError(f'Prior full verification seal is absent: {seal_path}')
        seal = read(seal_path)
        if seal.get('passed') is not True:
            raise RuntimeError(f'Prior verification seal did not pass: {seal_path}')


def require_predecessors():
    # No CUDA query or output creation occurs before actual terminal/seal checks.
    check_predecessor_records()
    for _, seal_path in PRIOR:
        seal = read(seal_path)
        for key in ('sources', 'postprocess_sources', 'artifact_sha256'):
            if not isinstance(seal.get(key), dict) or not seal[key]:
                raise ValueError(f'Incomplete prior verification binding: {seal_path}/{key}')
            check_bindings(seal[key])
        check_bindings(seal.get('inputs', {}))
    if read(PRIOR[1][1])['parent_verification_sha256'] != digest(PRIOR[0][1]):
        raise ValueError('P3 seal refers to a different HR verification')
    require_quiet_host()


def progress(path, value, retries=4, delay=.1):
    """Only noncritical telemetry may be skipped on a transient Windows reader lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    try:
        for attempt in range(retries):
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                if attempt+1 < retries:
                    time.sleep(delay)
        return False
    finally:
        if temporary.exists():
            temporary.unlink()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    return dict(path=str(path), sha256=digest(path))


def state_content(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(f'{name}|{a.dtype.str}|{a.shape}|'.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _load(path, expected_sha, device):
    checked(path, expected_sha)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('width') != WIDTH or not isinstance(payload.get('state_dict'), dict):
        raise ValueError('Wrong Seg1/Seg2 checkpoint schema or width')
    for value in payload['state_dict'].values():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Checkpoint has missing or nonfinite FP32 tensors')
    model = SkinUNet(width=WIDTH)
    model.load_state_dict(payload['state_dict'], strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError('Model parameter count changed')
    if any(not torch.equal(value, model.state_dict()[name]) for name, value in payload['state_dict'].items()):
        raise ValueError('Initialization weights are not exact')
    return model.to(device).eval()


def load_initial(path=INITIAL, expected_sha=INITIAL_SHA, device='cpu'):
    return _load(path, expected_sha, device)


def load_checkpoint(descriptor, device='cpu'):
    return _load(descriptor['path'], descriptor['sha256'], device)


def save_checkpoint(path, model, metadata):
    path = Path(path)
    if model.width != WIDTH:
        raise ValueError('Only the registered Seg2 model may be exported')
    payload = dict(metadata)
    payload.update(width=WIDTH, initial_sha256=INITIAL_SHA,
                   state_dict={k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()})
    if any(not torch.isfinite(v).all() for v in payload['state_dict'].values()):
        raise ValueError('Refusing nonfinite model export')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(payload, stream)
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size,
                state_content_sha256=state_content(model))


def tensor_batch(rgb, labels, device):
    if (rgb.dtype != np.uint8 or labels.dtype != np.uint8 or rgb.shape[:-1] != labels.shape
            or rgb.ndim != 4 or rgb.shape[-1] != 3 or np.any(labels > 1)):
        raise ValueError('Matching uint8 RGB and binary labels required')
    x = torch.from_numpy(np.array(rgb, copy=True)).permute(0,3,1,2)
    x = x.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)/255
    y = torch.from_numpy(np.array(labels, copy=True))[:,None].to(device=device, dtype=torch.float32)
    return x, y


def precision(device):
    return torch.autocast('cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()


def update(model, optimizer, scaler, x, y, step, trace=False):
    if not isinstance(step, int) or not 0 <= step < BUDGETS[-1]:
        raise ValueError('Update step outside the fixed schedule')
    factor = .1+.9*.5*(1+math.cos(math.pi*step/(BUDGETS[-1]-1)))
    for group in optimizer.param_groups:
        group['lr'] = .0001*factor
    model.train()
    x,y = augment(x,y)
    augmentation_sha = None
    if trace:
        h = hashlib.sha256()
        for value in (x[:len(x)//2], y[:len(y)//2]):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        augmentation_sha = h.hexdigest()
    optimizer.zero_grad(set_to_none=True)
    with precision(x.device):
        loss = skin_loss(model(x),y)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite training loss')
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    if any(p.grad is None for p in model.parameters()):
        raise ValueError('Missing training gradient')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    result = dict(loss=float(loss.detach()), gradient_norm=float(norm), learning_rate=.0001*factor,
                  successful_update=True)
    if trace:
        result['anchor_augmentation_sha256'] = augmentation_sha
    return result


def load_training_views():
    splits = {i:load_split(source, 'train') for i,source in enumerate(SOURCES)}
    if any(len(split.indices) != COUNTS[split.source]['train'] for split in splits.values()):
        raise ValueError('Training source sizes differ from registration')
    return splits


def setup(device):
    torch.set_num_threads(2)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def preflight(device):
    if device not in ('cpu', 'cuda'):
        raise ValueError('Unregistered preflight device')
    if device == 'cuda':
        require_predecessors()
    elif torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight must run without a CUDA context')
    registration = verify_registration()
    registration_sha = digest(RUN / 'registration.json')
    attempt = RUN / 'preflights' / f'{device}_{uuid.uuid4().hex}'
    receipt = dict(status='running', device=device, pid=os.getpid(), started_utc=utc(),
                   registration_sha256=registration_sha, batch_size=2 if device == 'cpu' else BATCH,
                   steps_per_case=2, cases=[], measured_quality=False, benchmark=False)
    write_once(attempt / 'started.json', receipt)
    started = time.perf_counter()
    try:
        setup(device)
        splits = load_training_views()
        anchors = {}
        for trajectory in registration['recipe']['trajectories']:
            model = load_initial(device=device).to(memory_format=torch.channels_last)
            initial_content = state_content(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
            scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
            sampler = PairedSampler(splits[0].indices, splits[1].indices, trajectory['seed'], receipt['batch_size'])
            torch.manual_seed(trajectory['seed'])
            case = dict(arm=trajectory['arm'], seed=trajectory['seed'], initial_content_sha256=initial_content,
                        steps=[], parameters=sum(p.numel() for p in model.parameters()))
            for step in range(2):
                sources, ids = sampler.next(trajectory['arm'])
                x,y = tensor_batch(*assemble_batch(splits, sources, ids), device)
                result = update(model, optimizer, scaler, x,y,step,trace=True)
                result.update(source_ids=sources.tolist(), global_ids=ids.tolist())
                case['steps'].append(result)
                key = (trajectory['seed'], step)
                paired = (ids[:len(ids)//2].tolist(), result['anchor_augmentation_sha256'])
                if trajectory['arm'] == 'lapa_only':
                    anchors[key] = paired
                elif anchors[key] != paired:
                    raise ValueError('Paired anchor samples or augmentation diverged')
            case['weights_changed'] = state_content(model) != initial_content
            if not case['weights_changed']:
                raise ValueError('Preflight failed to change model weights')
            receipt['cases'].append(case)
            print('SEG2 PREFLIGHT CASE', trajectory['id'], 'passed', flush=True)
            del model, optimizer, scaler, x,y
        if len({c['initial_content_sha256'] for c in receipt['cases']}) != 1:
            raise ValueError('Preflight initializations differ')
        check_bindings(registration['bindings'])
        receipt.update(status='passed', successful_updates=12, anchor_pairs_exact=True,
                       seconds=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized(),
                       torch_version=str(torch.__version__), finished_utc=utc())
        if device == 'cuda':
            receipt['device_name'] = torch.cuda.get_device_name(0)
            receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            receipt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        elif receipt['cuda_initialized']:
            raise ValueError('CPU preflight initialized CUDA')
        write_once(attempt / 'result.json', receipt)
        print('SEG2 PREFLIGHT PASSED', str(attempt / 'result.json'), flush=True)
        return receipt
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}',
                       seconds=time.perf_counter()-started, finished_utc=utc())
        write_once(attempt / 'failure.json', receipt)
        raise


def matching_preflight(device, registration_sha):
    candidates = []
    for path in sorted((RUN / 'preflights').glob(f'{device}_*/result.json')):
        value = read(path)
        if (value.get('status') == 'passed' and value.get('device') == device
                and value.get('registration_sha256') == registration_sha
                and value.get('successful_updates') == 12 and len(value.get('cases', [])) == 6
                and value.get('anchor_pairs_exact') is True
                and process_alive(value.get('pid')) is False
                and value.get('batch_size') == (BATCH if device == 'cuda' else 2)):
            candidates.append(dict(path=str(path), sha256=digest(path)))
    if not candidates:
        raise RuntimeError(f'No successful {device} preflight for the current registration')
    return candidates[0]


def require_finished(path):
    value = read(path)
    if value.get('status') != 'complete' or process_alive(value.get('pid')) is not False:
        raise RuntimeError(f'Stage worker must be complete and confirmed dead: {path}')
    return value


def run():
    require_predecessors()
    registration = verify_registration()
    registration_sha = digest(RUN / 'registration.json')
    preflights = {d:matching_preflight(d, registration_sha) for d in ('cpu', 'cuda')}
    if (RUN / 'job.json').exists() or (RUN / 'completion.json').exists():
        raise FileExistsError('Seg2 production exists; recovery must be separately documented')
    write_once(RUN / 'job.json', dict(status='running', pid=os.getpid(), started_utc=utc(),
                                     registration_sha256=registration_sha, preflights=preflights))
    started = time.perf_counter()
    try:
        # Lazy import keeps the TEST consumer out of CPU preflight execution.
        from skin_face_transfer_evaluate import evaluate_split
        setup('cuda')
        training = load_training_views()
        validation = {source:load_split(source, 'validation') for source in SOURCES}
        baseline = load_initial(device='cuda').to(memory_format=torch.channels_last)
        baseline_ref = dict(path=str(INITIAL), sha256=INITIAL_SHA, bytes=INITIAL.stat().st_size,
                            state_content_sha256=state_content(baseline))
        baseline_metrics = {}
        for source, split in validation.items():
            metric, arrays = evaluate_split(baseline, split, BATCH, 'cuda', include_color=False)
            metric['arrays'] = save_npz(RUN / 'baseline' / f'{source}_validation.npz', **arrays)
            baseline_metrics[source] = metric
        zero = dict(step=0, checkpoint=baseline_ref, validation=baseline_metrics)
        write_once(RUN / 'baseline/validation.json', zero)
        del baseline
        histories, receipts = {}, []
        completed = 0
        for trajectory in registration['recipe']['trajectories']:
            directory = RUN / 'trajectories' / trajectory['id']
            model = load_initial(device='cuda').to(memory_format=torch.channels_last)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
            scaler = torch.amp.GradScaler('cuda')
            sampler = PairedSampler(training[0].indices, training[1].indices, trajectory['seed'], BATCH)
            torch.manual_seed(trajectory['seed'])
            history, losses = [zero], []
            ids_seen = np.empty((BUDGETS[-1], BATCH), dtype=np.int64)
            sources_seen = np.empty((BUDGETS[-1], BATCH), dtype=np.uint8)
            trajectory_started, last_progress = time.perf_counter(), time.perf_counter()
            for step in range(BUDGETS[-1]):
                sources, ids = sampler.next(trajectory['arm'])
                x,y = tensor_batch(*assemble_batch(training,sources,ids), 'cuda')
                result = update(model,optimizer,scaler,x,y,step)
                losses.append([result['loss'], result['gradient_norm'], result['learning_rate']])
                ids_seen[step], sources_seen[step] = ids, sources
                completed += 1
                if (step+1) % INTERVAL == 0:
                    descriptor = save_checkpoint(directory/f'step_{step+1:04d}.pt', model,
                        dict(arm=trajectory['arm'], seed=trajectory['seed'], step=step+1,
                             registration_sha256=registration_sha))
                    metrics = {}
                    for source, split in validation.items():
                        metric, arrays = evaluate_split(model,split,BATCH,'cuda',include_color=False)
                        metric['arrays'] = save_npz(directory/f'step_{step+1:04d}_{source}_validation.npz', **arrays)
                        metrics[source] = metric
                    record = dict(step=step+1, checkpoint=descriptor, validation=metrics,
                                  elapsed_seconds=time.perf_counter()-trajectory_started)
                    write_once(directory/f'step_{step+1:04d}.json', record)
                    history.append(record)
                    print('SEG2 VALIDATION', trajectory['id'], step+1,
                          {s:m['mean_image_iou'] for s,m in metrics.items()}, flush=True)
                if time.perf_counter()-last_progress >= 10 or step+1 == BUDGETS[-1]:
                    elapsed = time.perf_counter()-started
                    value = dict(status='training', pid=os.getpid(), trajectory=trajectory['id'],
                                 step=step+1, total_steps=BUDGETS[-1], completed_updates=completed,
                                 total_updates=registration['recipe']['new_updates'],
                                 elapsed_seconds=elapsed, images_seen=completed*BATCH,
                                 images_per_second=completed*BATCH/elapsed,
                                 estimated_remaining_seconds=elapsed/completed*(registration['recipe']['new_updates']-completed),
                                 training_loss=result['loss'], updated_utc=utc())
                    progress(RUN / 'progress.json', value)
                    print('SEG2 PROGRESS', json.dumps(value), flush=True)
                    last_progress = time.perf_counter()
            sample_trace = save_npz(directory/'training_trace.npz', global_ids=ids_seen, source_ids=sources_seen,
                                    loss_gradient_lr=np.asarray(losses, dtype=np.float64))
            write_once(directory/'history.json', history)
            record = dict(**trajectory, successful_updates=BUDGETS[-1], attempted_batches=BUDGETS[-1],
                          elapsed_seconds=time.perf_counter()-trajectory_started,
                          history_path=str(directory/'history.json'), history_sha256=digest(directory/'history.json'),
                          sample_trace=sample_trace, cross_stream_duplicate_presentations=sampler.cross_stream_duplicate_presentations,
                          initial_content_sha256=baseline_ref['state_content_sha256'])
            write_once(directory/'receipt.json', record)
            receipts.append(record)
            histories[trajectory['id']] = history
            del model, optimizer, scaler, x,y
        selection = freeze_choices(histories)
        selection.update(registration_sha256=registration_sha, created_utc=utc(),
                         histories={r['history_path']:r['history_sha256'] for r in receipts})
        check_bindings(registration['bindings'])
        write_once(RUN / 'selection.json', selection)
        completion = dict(status='complete', pid=os.getpid(), trajectories=receipts, successful_updates=completed,
                          registration_sha256=registration_sha, selection_sha256=digest(RUN / 'selection.json'),
                          elapsed_seconds=time.perf_counter()-started, finished_utc=utc(), test_accessed=False)
        write_once(RUN / 'completion.json', completion)
        progress(RUN / 'progress.json', dict(status='training_complete_pending_test', completed_updates=completed,
                                            total_updates=completed, updated_utc=utc()))
        print('SEG2 TRAINING COMPLETE', completed, flush=True)
    except BaseException as error:
        write_once(RUN / 'failure.json', dict(error=f'{type(error).__name__}: {error}', pid=os.getpid(),
                                             elapsed_seconds=time.perf_counter()-started, failed_utc=utc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['register', 'preflight-cpu', 'preflight-cuda', 'run'])
    args = parser.parse_args()
    if args.command == 'register':
        freeze_registration()
        print('SEG2 REGISTERED', digest(RUN/'registration.json'), flush=True)
    elif args.command.startswith('preflight-'):
        preflight(args.command.removeprefix('preflight-'))
    else:
        run()


if __name__ == '__main__':
    main()
