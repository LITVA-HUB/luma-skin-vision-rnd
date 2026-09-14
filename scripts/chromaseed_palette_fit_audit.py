"""Independent parameter, optimization-diagnostic and native-encoder compatibility audit."""
import json
import time

import numpy as np
import torch
from chromaseed_architecture_scale import Bank
from chromaseed_palette_data import OUT, read, save, sha, verify
from scipy.special import expit
from threadpoolctl import threadpool_limits
from torch.nn import functional as F


def encoder(theta, standardized):
    weight1 = theta[:6912].reshape(18, 384)
    bias1 = theta[6912:7296]
    weight2 = theta[7296:105600].reshape(384, 256)
    bias2 = theta[105600:105856]
    value = standardized@weight1+bias1
    value = value*expit(value)
    value = value@weight2+bias2
    return value*expit(value)


def main():
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only audit required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    contract = dict(source_sha256=sha(__file__), source_lock_sha256=sha(OUT / 'source_lock.json'),
                    fit_sha256=sha(OUT / 'fit.json'), data_audit_sha256=sha(OUT / 'data_audit.json'),
                    mse_atol=1e-5, mse_rtol=1e-5, encoder_atol=2e-5, encoder_rtol=1e-6,
                    full_training_diagnostics=True, native_data_read=False)
    save(OUT / 'fit_audit_protocol.json', contract)
    verify()
    if not read(OUT / 'data_audit.json')['passed']:
        raise ValueError('Data audit has not passed')
    fit = read(OUT / 'fit.json')
    for name, field in [('data.npz', 'data_sha256'), ('aux_bank_2048.npz', 'auxiliary_bank_sha256'),
                        ('export_probe.npz', 'export_probe_sha256')]:
        if sha(OUT / name) != fit[field]:
            raise ValueError('Fit artifact changed: ' + name)
    z = dict(np.load(OUT / 'data.npz', allow_pickle=False))
    b = dict(np.load(OUT / 'aux_bank_2048.npz', allow_pickle=False))
    probe = dict(np.load(OUT / 'export_probe.npz', allow_pickle=False))
    if b['theta'].shape != (6, 115108) or b['initial_theta'].shape != (6, 115108) or fit['steps'] != 2048:
        raise ValueError('Incomplete auxiliary bank')
    np.testing.assert_array_equal(b['mean'], z['tokens'].astype(float).mean(0).astype(np.float32))
    np.testing.assert_array_equal(b['std'], np.maximum(z['tokens'].astype(float).std(0), 1e-6).astype(np.float32))
    np.testing.assert_array_equal(b['target_mean'], z['target'].mean(0).astype(np.float32))
    np.testing.assert_array_equal(b['target_std'], np.maximum(z['target'].std(0), 1e-6).astype(np.float32))
    draws = np.stack([np.random.default_rng(s+770003).integers(0, 19456, (2048, 128)) for s in (17, 29, 43)])
    np.testing.assert_array_equal(b['sampling'], draws)
    if [r['step'] for r in fit['history']] != list(range(128, 2049, 128)):
        raise ValueError('Incomplete fixed-step history')
    losses, maximum, compatibility = {}, 0., []
    with threadpool_limits(limits=1):
        # Full dataset arithmetic; this is training loss, not a held endpoint.
        standardized = (z['tokens'].astype(float)-b['mean'])/b['std']
        targets = [(z[k]-b['target_mean'])/b['target_std'] for k in ('target', 'shuffled_target')]
        for state_key, metric_key in [('initial_theta', 'initial_full_train_mse'), ('theta', 'final_full_train_mse')]:
            values = []
            for slot in range(6):
                theta = b[state_key][slot].astype(float)
                prediction = encoder(theta, standardized)@theta[105856:115072].reshape(256, 36)+theta[115072:]
                values.append(float(np.mean((prediction-targets[slot % 2])**2)))
            np.testing.assert_allclose(values, fit[metric_key], atol=contract['mse_atol'], rtol=contract['mse_rtol'])
            losses[metric_key] = values
        raw = z['tokens'][probe['row_indices']].astype(float)
        mean2 = (b['mean']+.2*b['std']).astype(np.float32)
        std2 = (b['std']*1.3).astype(np.float32)
        normalized2 = (raw-mean2)/std2
        transferred = []
        for slot, item in enumerate(fit['exports']):
            if sha(item['path']) != item['sha256']:
                raise ValueError('Export changed')
            model = dict(np.load(item['path'], allow_pickle=False))
            np.testing.assert_array_equal(model['theta'], b['theta'][slot, :105856])
            np.testing.assert_array_equal(model['mean'], b['mean'])
            np.testing.assert_array_equal(model['std'], b['std'])
            original = encoder(model['theta'].astype(float), (raw-b['mean'])/b['std'])
            np.testing.assert_allclose(original, probe['torch_outputs'][slot],
                                       atol=contract['encoder_atol'], rtol=contract['encoder_rtol'])
            maximum = max(maximum, float(np.max(np.abs(original-probe['torch_outputs'][slot]))))
            theta = model['theta'].astype(float)
            weight = theta[:6912].reshape(18, 384).copy()
            theta[:6912] = (weight*(std2.astype(float)/b['std'])[:, None]).ravel()
            theta[6912:7296] += ((mean2.astype(float)-b['mean'])/b['std'])@weight
            transferred.append(theta.astype(np.float32))
            np.testing.assert_allclose(encoder(theta.astype(np.float32).astype(float), normalized2), original,
                                       atol=contract['encoder_atol'], rtol=contract['encoder_rtol'])
        for variant in ('patch5m', 'soft5m', 'dynamic5m'):
            native = Bank(variant).eval()
            np.testing.assert_array_equal(native.theta.detach().numpy()[:, :105856], b['initial_theta'][:, :105856])
            with torch.inference_mode():
                native.theta[:, :105856].copy_(torch.from_numpy(np.asarray(transferred)))
                inp = torch.tensor(normalized2, dtype=torch.float32)[None].expand(6, -1, -1)
                out = F.silu(native.layers['token2'](F.silu(native.layers['token1'](inp)))).numpy()
            np.testing.assert_allclose(out, probe['torch_outputs'], atol=contract['encoder_atol'], rtol=contract['encoder_rtol'])
            drift = float(np.max(np.abs(out-probe['torch_outputs'])))
            compatibility.append(dict(variant=variant, six_encoders_transferred=True, maximum_drift=drift))
            del native
    if torch.cuda.is_initialized() or sha(__file__) != contract['source_sha256']:
        raise ValueError('Audit scope/source changed')
    result = dict(passed=True, audit_protocol_sha256=sha(OUT / 'fit_audit_protocol.json'),
                  full_auxiliary_models_checked=6, encoder_exports_checked=6, samples_per_training_loss_check=19456,
                  native_architecture_compatibility=compatibility, independent_losses=losses,
                  maximum_encoder_numpy_difference=maximum, sampling_hash_verified=True,
                  cuda_initialized=False, native_finetuning_performed=False,
                  native_accuracy_claim=False, seconds=time.perf_counter()-started)
    save(OUT / 'fit_audit.json', result)
    print('PALETTE FIT AUDIT COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
