"""Original spatial-source and photometric-view readback before auxiliary fitting."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from chromaseed_palette_data import OUT, P1, ROOT, read, save, sha, verify
from PIL import Image
from scipy.io import loadmat
from skin_mskcc_pixels import features
from skin_spectral_palette_audit import reference_integration
from threadpoolctl import threadpool_limits


def statistics(rgb):
    z = rgb.astype(np.float64)/255
    out = []
    for patch in z:
        flat = patch.reshape(256, 3)
        quantiles = np.percentile(flat, [10, 50, 90], axis=0).reshape(-1)
        average = flat.sum(0)/256
        deviation = np.sqrt(((flat-average)**2).sum(0)/256)
        vertical = np.abs(patch[1:]-patch[:-1]).sum((0, 1))/(15*16)
        horizontal = np.abs(patch[:, 1:]-patch[:, :-1]).sum((0, 1))/(16*15)
        out.append(np.r_[quantiles, average, deviation, .5*(vertical+horizontal)])
    return np.asarray(out)


def main():
    started = time.perf_counter()
    contract = dict(source_sha256=sha(__file__), source_lock_sha256=sha(OUT / 'source_lock.json'),
                    data_sha256=sha(OUT / 'data.npz'), profile_sha256=sha(OUT / 'data_profile.json'),
                    reference_integration_source_sha256=sha(ROOT / 'scripts/skin_spectral_palette_audit.py'),
                    reflectance_atol=1e-12, rgb_atol=1e-12, lab_atol=1e-8, token_atol=1e-7, rtol=0.)
    save(OUT / 'data_audit_protocol.json', contract)
    lock = verify()
    profile = read(OUT / 'data_profile.json')
    if contract['data_sha256'] != profile['data_sha256']:
        raise ValueError('Data checksum mismatch')
    z = dict(np.load(OUT / 'data.npz', allow_pickle=False))
    p1 = dict(np.load(P1 / 'palette.npz', allow_pickle=False))
    masks = read(P1 / 'profile.json')['files']
    rgb_xyz = np.array([[.4124564, .3575761, .1804375], [.2126729, .7151522, .0721750],
                        [.0193339, .1191920, .9503041]])
    maximum_tokens, maximum_rgb, maximum_mean, clipping = 0., 0., 0., 0
    with threadpool_limits(limits=1):
        for group, source in enumerate(lock['sources']):
            if source['role'] != 'train':
                raise ValueError('Held source in auxiliary data')
            mask = np.asarray(Image.open(masks[group]['previews']['mask'])) > 0
            original_centres = p1['pixel_yx'][p1['source_index'] == group]
            expected = []
            for y, x in original_centres:
                if 8 <= y <= mask.shape[0]-8 and 8 <= x <= mask.shape[1]-8:
                    if np.count_nonzero(mask[y-8:y+8, x-8:x+8]) == 256:
                        expected.append((y, x))
            rows = np.flatnonzero(z['patch_source'] == group)
            np.testing.assert_array_equal(z['patch_centres_yx'][rows], np.array(expected[:128]))
            if len(rows) != 128:
                raise ValueError('Unequal source weighting')
            cube = loadmat(source['path'], variable_names=['datao'], verify_compressed_data_integrity=True)['datao']
            for patch_id in rows:
                y, x = z['patch_centres_yx'][patch_id]
                raw = cube[y-8:y+8, x-8:x+8].reshape(256, 33)
                mean = raw.sum(0)/256
                maximum_mean = max(maximum_mean, float(np.max(np.abs(mean-z['patch_mean_spectrum'][patch_id]))))
                np.testing.assert_allclose(mean, z['patch_mean_spectrum'][patch_id], rtol=0, atol=contract['reflectance_atol'])
                linear = np.linalg.solve(rgb_xyz, (raw@p1['matrix_2deg']).T).T.reshape(16, 16, 3)
                maximum_rgb = max(maximum_rgb, float(np.max(np.abs(linear-z['patch_linear_rgb'][patch_id]))))
                np.testing.assert_allclose(linear, z['patch_linear_rgb'][patch_id], rtol=0, atol=contract['rgb_atol'])
                parameters = np.ones((8, 5))
                for view in range(1, 8):
                    key = f'LumaPalettePretrainV1|{source["sha256"]}|{y}|{x}|{view}'
                    seed = int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big')
                    rng = np.random.default_rng(seed)
                    parameters[view] = np.r_[np.exp(rng.uniform(-.25, .25, 3)),
                                              2**rng.uniform(-.5, .5), rng.uniform(.85, 1.15)]
                np.testing.assert_array_equal(parameters, z['view_parameters'][patch_id])
                perturbed = linear[None]*parameters[:, None, None, :3]*parameters[:, None, None, 3:4]
                encoded = np.empty_like(perturbed)
                low = perturbed <= .0031308
                encoded[low] = perturbed[low]*12.92
                encoded[~low] = 1.055*perturbed[~low]**(1/2.4)-.055
                clipping += int(((encoded < 0) | (encoded > 1)).sum())
                rendered = np.rint(np.clip(encoded, 0, 1)**parameters[:, None, None, 4:5]*255).astype(np.uint8)
                tokens = statistics(rendered)
                maximum_tokens = max(maximum_tokens, float(np.max(np.abs(tokens-z['tokens'][patch_id*8:(patch_id+1)*8]))))
                np.testing.assert_allclose(tokens, z['tokens'][patch_id*8:(patch_id+1)*8], rtol=0, atol=contract['token_atol'])
                if patch_id == rows[0]:
                    native = features(np.tile(rendered[0], (8, 8, 1)))['tokens']
                    np.testing.assert_allclose(native, np.tile(tokens[0], (64, 1)), rtol=0, atol=1e-12)
            del cube
            print('PALETTE DATA AUDIT', group+1, '/ 19', '128 source crops exact', flush=True)
        _, lab, _ = reference_integration(z['patch_mean_spectrum'], p1['wavelength_nm'],
            p1['integration_wave_nm'], p1['cmf_10deg'], p1['d65_spd'])
        target = np.c_[z['patch_mean_spectrum'], lab]
        np.testing.assert_allclose(z['patch_target'], target, rtol=0, atol=contract['lab_atol'])
        np.testing.assert_array_equal(z['target'], np.repeat(z['patch_target'], 8, axis=0))
        permutation = np.random.default_rng(770019).permutation(2432)
        np.testing.assert_array_equal(z['patch_permutation'], permutation)
        np.testing.assert_array_equal(z['shuffled_target'], np.repeat(z['patch_target'][permutation], 8, axis=0))
        np.testing.assert_array_equal(z['source_index'], np.repeat(z['patch_source'], 8))
        np.testing.assert_array_equal(z['view_index'], np.tile(np.arange(8), 2432))
        np.testing.assert_array_equal(z['source_sha256'], p1['source_sha256'])
    if clipping != profile['clipped_rendered_channels'] or sha(__file__) != contract['source_sha256']:
        raise ValueError('Audit count/source mismatch')
    result = dict(passed=True, audit_protocol_sha256=sha(OUT / 'data_audit_protocol.json'),
                  source_groups=19, original_spatial_patches=2432, view_features_checked=19456,
                  native_feature_order_probes=19, max_mean_reflectance_drift=maximum_mean,
                  max_linear_rgb_drift=maximum_rgb, max_token_drift=maximum_tokens,
                  max_derived_lab_drift=float(np.max(np.abs(lab-z['patch_target'][:, -3:]))),
                  shuffled_fixed_points=int((permutation == np.arange(2432)).sum()),
                  clipped_rendered_channels=clipping, held_sources_decoded=0,
                  seconds=time.perf_counter()-started)
    save(OUT / 'data_audit.json', result)
    print('PALETTE DATA AUDIT COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
