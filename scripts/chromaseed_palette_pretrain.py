"""Fixed CPU auxiliary fits; no native color accuracy is inferred from this task."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from chromaseed_palette_data import OUT, read, save, sha, verify
from chromaseed_palette_encoder import (
    ENCODER_PARAMETERS,
    SEEDS,
    SLOTS,
    EncoderBank,
    numpy_encode,
    transplant,
)
from chromaseed_refine import BankAdamW
from threadpoolctl import threadpool_limits


def fit_arrays(x, aligned, shuffled, steps=2048, progress=None):
    if (x.shape[1:] != (18,) or aligned.shape != shuffled.shape or aligned.shape != (len(x), 36)
            or not len(x) or not all(np.isfinite(a).all() for a in (x, aligned, shuffled)) or not 1 <= steps <= 2048):
        raise ValueError('Matching finite auxiliary TRAIN arrays required')
    if torch.cuda.is_initialized():
        raise ValueError('Pretraining must use CPU while HR is active')
    mean = x.astype(float).mean(0).astype(np.float32)
    std = np.maximum(x.astype(float).std(0), 1e-6).astype(np.float32)
    ym = aligned.astype(float).mean(0).astype(np.float32)
    ys = np.maximum(aligned.astype(float).std(0), 1e-6).astype(np.float32)
    xt = torch.tensor((x-mean)/std, dtype=torch.float32)
    yt = torch.tensor(np.stack(((aligned-ym)/ys, (shuffled-ym)/ys)), dtype=torch.float32)
    indices = np.stack([np.random.default_rng(seed+770003).integers(0, len(x), (steps, 128)) for seed in SEEDS])
    draw = torch.from_numpy(indices)
    pair = torch.tensor([0, 0, 1, 1, 2, 2])
    arm = torch.tensor([0, 1, 0, 1, 0, 1])
    net = EncoderBank()
    initial_theta = net.theta.detach().numpy().copy()
    opt = BankAdamW(net.parameters(), np.full(6, .001, np.float32), weight_decay=.01, max_norm=5.)

    def full_loss():
        with torch.inference_mode():
            errors = torch.zeros(6)
            for start in range(0, len(x), 256):
                output = net(xt[start:start+256][None].expand(6, -1, -1))
                target = yt[arm, start:start+256]
                errors += (output-target).square().mean(-1).sum(-1)
            return (errors/len(x)).tolist()

    initial_loss = full_loss()
    history = []
    started = time.perf_counter()
    for step in range(steps):
        ids = draw[:, step].index_select(0, pair)
        net.zero_grad(set_to_none=True)
        pred = net(xt[ids])
        target = yt[arm[:, None], ids]
        loss = (pred-target).square().mean((1, 2))
        if not torch.isfinite(loss).all():
            raise ValueError('Nonfinite auxiliary loss')
        loss.sum().backward()
        factor = np.float32(.1+.45*(1+np.cos(np.pi*step/2047)))
        opt.lrs.fill_(float(np.float32(.001)*factor))
        correction = torch.tensor([1-.9**(step+1), np.sqrt(1-.999**(step+1))], dtype=torch.float32)
        opt.step(correction)
        if (step+1) % 128 == 0 or step+1 == steps:
            item = dict(step=step+1, seconds=time.perf_counter()-started, minibatch_mse=loss.detach().tolist())
            history.append(item)
            if progress is not None:
                progress(item)
    seconds = time.perf_counter()-started
    final_loss = full_loss()
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    state = dict(theta=net.theta.detach().numpy().copy(), initial_theta=initial_theta,
                 mean=mean, std=std, target_mean=ym, target_std=ys, sampling=indices)
    info = dict(steps=steps, batch_size=128, slots=[list(s) for s in SLOTS],
                initial_full_train_mse=initial_loss, final_full_train_mse=final_loss,
                history=history, fitting_seconds=seconds, cuda_initialized=False,
                trainable_parameters_per_model=115108, exported_encoder_parameters=ENCODER_PARAMETERS)
    return net, state, info


def main():
    started = time.perf_counter()
    if (OUT / 'fit.json').exists() or (OUT / 'aux_bank_2048.npz').exists():
        raise ValueError('Do not overwrite auxiliary fits')
    lock = verify()
    profile = read(OUT / 'data_profile.json')
    if sha(OUT / 'data.npz') != profile['data_sha256']:
        raise ValueError('Prepared data changed')
    if lock['steps'] != 2048:
        raise ValueError('Fixed training budget changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = dict(np.load(OUT / 'data.npz', allow_pickle=False))
    save(OUT / 'fit_start.json', dict(pid=os.getpid(), source_lock_sha256=sha(OUT / 'source_lock.json'),
                                     data_profile_sha256=sha(OUT / 'data_profile.json'), device='cpu'))

    def progress(item):
        # Status is stdout only: a dashboard reader cannot make this fit fail.
        print('PALETTE PRETRAIN', json.dumps(item), flush=True)

    with threadpool_limits(limits=1):
        net, state, info = fit_arrays(data['tokens'], data['target'], data['shuffled_target'], progress=progress)
        np.savez_compressed(OUT / 'aux_bank_2048.npz', **state)
        rows = np.random.default_rng(770057).choice(len(data['tokens']), 512, replace=False)
        raw = data['tokens'][rows]
        xx = torch.tensor((raw-state['mean'])/state['std'], dtype=torch.float32)[None].expand(6, -1, -1)
        with torch.inference_mode():
            reference = net.encode(xx).numpy()
        exported = []
        for slot, (seed, arm) in enumerate(SLOTS):
            model = net.export(slot, state['mean'], state['std'])
            path = OUT / f'encoder_s{seed}_{arm}.npz'
            np.savez_compressed(path, **model)
            independent = numpy_encode(model, raw)
            np.testing.assert_allclose(independent, reference[slot], atol=2e-5, rtol=1e-6)
            # Synthetic alternate normalization validates transfer algebra, not
            # the statistics of any future native fitting cohort.
            alternative = transplant(model, state['mean']+.2*state['std'], state['std']*1.3)
            transferred = numpy_encode(alternative, raw)
            np.testing.assert_allclose(independent, transferred, atol=2e-5, rtol=1e-6)
            exported.append(dict(seed=seed, arm=arm, path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                                 max_encoder_numpy_drift=float(np.max(np.abs(independent-reference[slot]))),
                                 max_transfer_drift=float(np.max(np.abs(independent-transferred)))))
        np.savez_compressed(OUT / 'export_probe.npz', row_indices=rows, torch_outputs=reference)
    verify()
    info.update(source_lock_sha256=sha(OUT / 'source_lock.json'), data_sha256=sha(OUT / 'data.npz'),
                auxiliary_bank_sha256=sha(OUT / 'aux_bank_2048.npz'), exports=exported,
                export_probe_sha256=sha(OUT / 'export_probe.npz'), comparison_atol=2e-5, comparison_rtol=1e-6,
                complete_workflow_seconds=time.perf_counter()-started,
                native_finetuning_performed=False, native_color_quality_gain_measured=False)
    save(OUT / 'fit.json', info)
    print('PALETTE PRETRAIN COMPLETE', json.dumps({k: v for k, v in info.items() if k not in ('history', 'exports')}), flush=True)


if __name__ == '__main__':
    main()
