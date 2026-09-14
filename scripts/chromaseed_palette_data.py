"""Spatial measured patches and fixed photometric simulations for encoder pretraining."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/Luma-RnD/chromaseed_palette_pretrain_v1')
P1 = Path('D:/Luma-RnD/data_growth_2026_09_14/uminho_palette_v1')
P1_PALETTE_SHA = '9e5bb25250258fb79a547b3ac6dc93792d1808967fe5842e21b42f57a6cc2112'
PROTOCOL = ROOT / 'docs/research/chromaseed_palette_pretrain_v1_protocol.md'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def patch_tokens(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 4 or rgb.shape[1:] != (16, 16, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 N x 16 x 16 x 3 patches')
    z = rgb.astype(np.float64)/255
    flat = z.reshape(len(z), 256, 3)
    quant = np.quantile(flat, [.1, .5, .9], axis=1).transpose(1, 0, 2).reshape(len(z), 9)
    gradient = .5*np.abs(np.diff(z, axis=1)).mean((1, 2))+.5*np.abs(np.diff(z, axis=2)).mean((1, 2))
    return np.concatenate((quant, flat.mean(1), flat.std(1), gradient), axis=1)


def spatial_centres(mask, centres, cap=128):
    mask, centres = np.asarray(mask), np.asarray(centres)
    if mask.ndim != 2 or mask.dtype != bool or centres.ndim != 2 or centres.shape[1] != 2 or cap < 1:
        raise ValueError('Expected binary source mask and pixel centres')
    chosen = []
    for y, x in centres:
        if y >= 8 and x >= 8 and y+8 <= mask.shape[0] and x+8 <= mask.shape[1]:
            if mask[y-8:y+8, x-8:x+8].all():
                chosen.append((y, x))
                if len(chosen) == cap:
                    return np.asarray(chosen, np.int32)
    raise ValueError(f'Only {len(chosen)} complete masked patches; {cap} required')


def view_parameters(source_sha, y, x):
    result = np.ones((8, 5), np.float64)
    for view in range(1, 8):
        key = f'LumaPalettePretrainV1|{source_sha}|{y}|{x}|{view}'
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest(), 'big'))
        result[view, :3] = np.exp(rng.uniform(-.25, .25, 3))
        result[view, 3] = 2**rng.uniform(-.5, .5)
        result[view, 4] = rng.uniform(.85, 1.15)
    return result


def rendered_views(linear_rgb, parameters):
    sys.path.insert(0, str(ROOT / 'src'))
    from luma_skin_vision.color import linear_to_srgb

    rgb = linear_rgb[None]*parameters[:, None, None, :3]*parameters[:, None, None, 3:4]
    encoded = linear_to_srgb(rgb)
    clipped_channels = int(((encoded < 0) | (encoded > 1)).sum())
    rendered = np.clip(encoded, 0, 1)**parameters[:, None, None, 4:5]
    return np.rint(rendered*255).astype(np.uint8), clipped_channels


def freeze():
    p1lock, profile = read(P1 / 'protocol.json'), read(P1 / 'profile.json')
    if sha(P1 / 'palette.npz') != P1_PALETTE_SHA or not read(P1 / 'audit.json')['status'] == 'passed':
        raise ValueError('Expected verified P1 data')
    paths = [Path(__file__), PROTOCOL, ROOT / 'scripts/chromaseed_palette_encoder.py',
             ROOT / 'scripts/chromaseed_palette_pretrain.py', ROOT / 'tests/test_chromaseed_palette_pretrain.py',
             ROOT / 'scripts/chromaseed_architecture_scale.py', ROOT / 'scripts/chromaseed_refine.py',
             ROOT / 'scripts/skin_mskcc_pixels.py',
             P1 / 'protocol.json', P1 / 'palette.npz', P1 / 'profile.json',
             P1 / 'audit_protocol.json', P1 / 'audit.json', P1 / 'visual_review.json']
    bindings = dict(p1lock['bindings'])
    for row in p1lock['sources']:
        bindings[row['path']] = row['sha256']
    for row in profile['files']:
        bindings[row['previews']['mask']] = row['preview_sha256']['mask']
    bindings.update({str(p): sha(p) for p in paths})
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Changed preparation input: ' + path)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'source_lock.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), bindings=bindings,
         p1_palette_sha256=P1_PALETTE_SHA, sources=p1lock['sources'], parameters_per_aux_model=115108,
         encoder_parameters=105856, patches_per_source=128, views_per_patch=8, steps=2048, device='cpu'))
    print('PALETTE PRETRAIN FROZEN', sha(OUT / 'source_lock.json'), flush=True)


def verify():
    lock = read(OUT / 'source_lock.json')
    for path, expected in lock['bindings'].items():
        if sha(path) != expected:
            raise ValueError('Frozen dependency changed: ' + path)
    return lock


def prepare():
    from PIL import Image
    from scipy.io import loadmat
    from skin_spectral_palette import xyz_lab
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(ROOT / 'src'))
    from luma_skin_vision.color import RGB_XYZ

    if (OUT / 'data.npz').exists() or (OUT / 'data_profile.json').exists():
        raise ValueError('Do not overwrite prepared inputs')
    started = time.perf_counter()
    lock = verify()
    z = dict(np.load(P1 / 'palette.npz', allow_pickle=False))
    profile = read(P1 / 'profile.json')
    means, rgbs, centres, groups, tokens, params = [], [], [], [], [], []
    clipping = 0
    with threadpool_limits(limits=1):
        for i, (source, record) in enumerate(zip(lock['sources'], profile['files'], strict=True)):
            mask = np.asarray(Image.open(record['previews']['mask'])) > 0
            selected = spatial_centres(mask, z['pixel_yx'][z['source_index'] == i])
            cube = loadmat(source['path'], variable_names=['datao'], verify_compressed_data_integrity=True)['datao']
            if cube.dtype != np.float64 or not np.isfinite(cube).all():
                raise ValueError('Expected unchanged original spectra')
            for y, x in selected:
                patch = cube[y-8:y+8, x-8:x+8]
                if patch.shape != (16, 16, 33) or not (patch > 0).all():
                    raise ValueError('Source support mismatch')
                mean = patch.mean((0, 1))
                rgb = (patch @ z['matrix_2deg']) @ np.linalg.inv(RGB_XYZ).T
                parameter = view_parameters(source['sha256'], int(y), int(x))
                rendered, clipped = rendered_views(rgb, parameter)
                clipping += clipped
                means.append(mean)
                rgbs.append(rgb)
                centres.append((y, x))
                groups.append(i)
                tokens.append(patch_tokens(rendered).astype(np.float32))
                params.append(parameter)
            del cube
            print('PALETTE SPATIAL SOURCE', i+1, '/ 19', len(selected), 'patches', flush=True)
    mean_spectra = np.asarray(means)
    lab = xyz_lab(mean_spectra @ z['matrix_10deg'], z['white_10deg'])
    targets = np.concatenate((mean_spectra, lab), axis=1)
    permutation = np.random.default_rng(770019).permutation(len(means))
    values = dict(tokens=np.concatenate(tokens), target=np.repeat(targets, 8, axis=0),
                  shuffled_target=np.repeat(targets[permutation], 8, axis=0),
                  patch_mean_spectrum=mean_spectra, patch_target=targets, patch_permutation=permutation,
                  patch_linear_rgb=np.asarray(rgbs), patch_centres_yx=np.asarray(centres, np.int32),
                  patch_source=np.asarray(groups, np.int16), source_index=np.repeat(groups, 8),
                  view_parameters=np.asarray(params), view_index=np.tile(np.arange(8), len(means)),
                  source_sha256=z['source_sha256'], wavelength_nm=z['wavelength_nm'])
    if values['tokens'].shape != (19456, 18) or targets.shape != (2432, 36):
        raise ValueError('Incomplete spatial dataset')
    np.savez_compressed(OUT / 'data.npz', **values)
    result = dict(source_lock_sha256=sha(OUT / 'source_lock.json'), data_sha256=sha(OUT / 'data.npz'),
                  bytes=(OUT / 'data.npz').stat().st_size, sources=19, spatial_patches=2432, examples=19456,
                  views_per_patch=8, clipped_rendered_channels=clipping,
                  total_rendered_channels=19456*256*3, reflectance_clipped=False,
                  patch_mean_above_one_channels=int((mean_spectra > 1).sum()),
                  new_camera_photographs=0, held_sources_decoded=0,
                  simulated_photometric_conditions=True, seconds=time.perf_counter()-started)
    verify()
    save(OUT / 'data_profile.json', result)
    print('PALETTE SPATIAL COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'verify'])
    action = parser.parse_args().action
    {'freeze': freeze, 'prepare': prepare, 'verify': verify}[action]()
