"""Measured TRAIN spectra with Seg1 sampling and explicitly derived D65 colors."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/Luma-RnD/data_growth_2026_09_14')
OUT = DATA / 'uminho_palette_v1'
LEGACY = Path('C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/public/uminho_hsfd_v1')
PROFILE = DATA / 'uminho_train_v2/profile.json'
PROFILE_SHA = '1fc862ebaeb65ae46528fb9a4ad33fe90f90f858dab6bfc355388abf40b4e226'
MANIFEST_SHA = '3d3b7dc7256eb04250dee2f5ef76880f4101d019313d97ba051480f8d320b5c7'
CHECKPOINT = DATA / 'facial_skin_v1/best.pt'
CHECKPOINT_SHA = '1203cbc5ed2ee17cb2a408c47a23b3a28468f169d48b4d3cee8bd3059e7fbed3'
PROTOCOL = ROOT / 'docs/research/skin_spectral_palette_v1_protocol.md'
CIE2 = ROOT / 'docs/data/provenance/skin_public_2026_09_11'
CIE10 = ROOT / 'docs/data/provenance/cie_material_v1'
CIE_FILES = [
    (CIE2 / 'CIE_xyz_1931_2deg.csv', CIE2 / 'CIE_xyz_1931_2deg.csv_metadata.json'),
    (CIE10 / 'CIE_xyz_1964_10deg.csv', CIE10 / 'CIE_xyz_1964_10deg.csv_metadata.json'),
    (CIE10 / 'CIE_std_illum_D65.csv', CIE10 / 'CIE_std_illum_D65.csv_metadata_v2.json'),
]


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def integration_matrix(knots, wave, cmf, spd):
    knots, wave, cmf, spd = [np.asarray(a, dtype=np.float64) for a in (knots, wave, cmf, spd)]
    if (knots.ndim != 1 or len(knots) < 2 or wave.ndim != 1 or len(wave) < 2
            or cmf.shape != (len(wave), 3) or spd.shape != wave.shape
            or not all(np.isfinite(a).all() for a in (knots, wave, cmf, spd))
            or not (np.diff(knots) > 0).all() or not (np.diff(wave) > 0).all()
            or (cmf < 0).any() or (spd < 0).any()):
        raise ValueError('Expected ordered wavelengths and finite, nonnegative CMF/SPD')
    weights = cmf * spd[:, None]
    if weights[:, 1].sum() <= 0:
        raise ValueError('Illuminant has no luminance support')
    # Production uses a uniform 1 nm grid. This is a normalized discrete sum,
    # with piecewise-linear reflectance and constant endpoint extrapolation.
    interpolation = np.stack([np.interp(wave, knots, row) for row in np.eye(len(knots))])
    return interpolation @ weights / weights[:, 1].sum()


def xyz_lab(xyz, white):
    xyz, white = np.asarray(xyz, dtype=np.float64), np.asarray(white, dtype=np.float64)
    if (xyz.shape[-1:] != (3,) or white.shape != (3,) or not (white > 0).all()
            or not np.isfinite(xyz).all() or not np.isfinite(white).all()):
        raise ValueError('Expected finite XYZ triples and a positive white point')
    ratio = xyz / white
    d = 6 / 29
    f = np.where(ratio > d**3, np.cbrt(ratio), ratio / (3*d*d) + 4/29)
    return np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]),
                     200*(f[..., 1]-f[..., 2])], axis=-1)


def sample_indices(mask, source_sha):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool or len(source_sha) != 64:
        raise ValueError('Expected a 2D boolean mask and source SHA256')
    available = np.flatnonzero(mask)
    if len(available) < 16:
        return np.empty(0, dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(('LumaSpectralPaletteV1|' + source_sha).encode()).digest(), 'big')
    return np.random.default_rng(seed).choice(available, min(1024, len(available)), replace=False)


def qualified_mask(logits, foreground):
    from PIL import Image
    from scipy.ndimage import binary_erosion

    logits, foreground = np.asarray(logits), np.asarray(foreground)
    if (logits.ndim != 2 or not np.isfinite(logits).all() or foreground.ndim != 2
            or foreground.dtype != bool):
        raise ValueError('Expected finite 2D logits and boolean source support')
    interior = binary_erosion(logits >= np.log(19.), structure=np.ones((3, 3), bool),
                              iterations=2, border_value=0)
    large = Image.fromarray(interior.astype(np.uint8)*255).resize(
        (foreground.shape[1], foreground.shape[0]), Image.Resampling.NEAREST)
    return (np.asarray(large) > 0) & foreground


def checked_profile():
    if digest(PROFILE) != PROFILE_SHA or digest(LEGACY / 'manifest.json') != MANIFEST_SHA:
        raise ValueError('Frozen source allocation/profile changed')
    profile = read_json(PROFILE)
    train = sorted(r['name'] for r in read_json(LEGACY / 'manifest.json')['rows'] if r['role'] == 'train')
    rows = profile['files']
    if (len(rows) != 19 or len({r['sha256'] for r in rows}) != 19
            or [Path(r['path']).name for r in rows] != train
            or any(r['role'] != 'train' for r in rows) or profile['held_cubes_acquired']):
        raise ValueError('Exactly 19 distinct original TRAIN cubes required')
    for row in rows:
        path = Path(row['path'])
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Source cube changed: ' + str(path))
    return rows


def original_cie():
    for table, metadata in CIE_FILES:
        for item in read_json(metadata)['checksums']:
            if digest(table, item['hashMethod']) != item['checksum']:
                raise ValueError('Original CIE table checksum mismatch')
    a, b, d65 = [np.genfromtxt(pair[0], delimiter=',') for pair in CIE_FILES]
    wave = np.arange(360., 831.)
    if not np.array_equal(a[:, 0], wave) or not np.array_equal(b[:, 0], wave):
        raise ValueError('CIE observer grid differs')
    if not np.array_equal(np.isnan(b[:, 3]), wave > 559):
        raise ValueError('Unexpected CIE1964 zbar support')
    b[wave > 559, 3] = 0.  # Metadata explicitly defines zero outside column support.
    if not all(np.isfinite(x).all() for x in (a, b, d65)):
        raise ValueError('Non-finite CIE data')
    spd = np.interp(wave, d65[:, 0], d65[:, 1])
    return wave, a[:, 1:], b[:, 1:], spd


def freeze():
    rows = checked_profile()
    original_cie()
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Seg1 checkpoint changed')
    paths = [Path(__file__), PROTOCOL, ROOT / 'tests/test_skin_spectral_palette.py',
             ROOT / 'scripts/skin_face_segment.py', ROOT / 'src/luma_skin_vision/color.py',
             PROFILE, LEGACY / 'manifest.json', LEGACY / 'skin_regions_v1.json',
             LEGACY / 'train_expand_v1.json', CHECKPOINT]
    paths += [p for pair in CIE_FILES for p in pair]
    lock = dict(protocol='skin_spectral_palette_v1',
                created_utc=datetime.now(timezone.utc).isoformat(),
                bindings={str(p): digest(p) for p in paths}, sources=rows,
                device='cpu', threads=1, input_size=192, logit_threshold=float(np.log(19.)),
                erosion_iterations=2, samples_per_source_cap=1024, source_minimum=16,
                native_color_target=False, held_sources_read=False,
                python=platform.python_version(), numpy=np.__version__)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / 'protocol.json', lock)
    print('PALETTE FROZEN', digest(OUT / 'protocol.json'), flush=True)


def verify_lock():
    lock = read_json(OUT / 'protocol.json')
    for path, expected in lock['bindings'].items():
        if digest(path) != expected:
            raise ValueError('Frozen dependency changed: ' + path)
    rows = checked_profile()
    if rows != lock['sources']:
        raise ValueError('Source list changed')
    return lock


def preview_tile(rgb, overlay, label):
    from PIL import Image, ImageDraw, ImageOps

    tile = Image.new('RGB', (544, 348), (22, 26, 33))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 8), label, fill=(235, 240, 245))
    draw.text((8, 329), 'Derived D65 RGB', fill=(220, 225, 230))
    draw.text((280, 329), 'Seg1 sampling mask', fill=(160, 240, 195))
    for i, array in enumerate((rgb, overlay)):
        pic = ImageOps.contain(Image.fromarray(array), (264, 300))
        tile.paste(pic, (i*272 + (272-pic.width)//2, 27 + (300-pic.height)//2))
    return tile


def prepare():
    import torch
    from PIL import Image
    from scipy.io import loadmat
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(ROOT / 'src'))
    from skin_face_segment import SkinUNet

    from luma_skin_vision.color import RGB_XYZ, linear_to_srgb

    if any((OUT / name).exists() for name in ('palette.npz', 'profile.json', 'previews')):
        raise ValueError('Preserve existing run; no overwrite or automatic retry')
    started = time.perf_counter()
    lock = verify_lock()
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only palette preparation required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if state['width'] != 24:
        raise ValueError('Frozen Seg1 width differs')
    model = SkinUNet(width=24).eval()
    model.load_state_dict(state['state_dict'], strict=True)
    knots = np.arange(400., 721., 10.)
    wave, cmf2, cmf10, spd = original_cie()
    matrices = [integration_matrix(knots, wave, cmf, spd) for cmf in (cmf2, cmf10)]
    whites = [m.sum(0) for m in matrices]
    outside = (wave < knots[0]) | (wave > knots[-1])
    missing = [(c*spd[:, None])[outside].sum(0)/(c[:, 1]*spd).sum() for c in (cmf2, cmf10)]
    inverse_rgb = np.linalg.inv(RGB_XYZ).T
    roi_rows = read_json(LEGACY / 'train_expand_v1.json')['rows']
    rois = {r['name']: boxes for r, boxes in zip(roi_rows, read_json(LEGACY / 'skin_regions_v1.json'), strict=True)}
    preview_dir = OUT / 'previews'
    preview_dir.mkdir()
    chunks, records, tiles = [], [], []
    with threadpool_limits(limits=1), torch.inference_mode():
        for index, row in enumerate(lock['sources']):
            cube = loadmat(row['path'], variable_names=['datao'],
                           verify_compressed_data_integrity=True)['datao']
            if (cube.ndim != 3 or cube.shape[-1] != 33 or cube.dtype != np.float64
                    or not np.isfinite(cube).all() or (cube < 0).any()):
                raise ValueError('Expected finite, nonnegative original float64 spectral cube')
            foreground = np.all(cube > 0, axis=-1)
            encoded = linear_to_srgb((cube @ matrices[0]) @ inverse_rgb)
            rgb = np.rint(np.clip(encoded, 0, 1)*255).astype(np.uint8)
            resized = np.asarray(Image.fromarray(rgb).resize((192, 192), Image.Resampling.BILINEAR))
            x = torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0).float()/255
            logits = model(x)[0, 0].numpy()
            mask = qualified_mask(logits, foreground)
            flat = sample_indices(mask, row['sha256'])
            y, xcoord = np.unravel_index(flat, foreground.shape)
            spectra = cube[y, xcoord].copy()
            n = len(spectra)
            chunks.append(dict(spectra=spectra, source_index=np.full(n, index, np.int16),
                               pixel_yx=np.stack((y, xcoord), axis=-1).astype(np.int32),
                               source_weight=np.full(n, 1/n if n else 0., dtype=np.float64)))
            tag = f'{index+1:02d}_{row["sha256"][:8]}'
            paths = {key: preview_dir / f'{tag}_{key}{suffix}' for key, suffix in
                     [('rgb', '.png'), ('mask', '.png'), ('overlay', '.png'), ('logits', '.npy')]}
            overlay = rgb.copy()
            overlay[mask] = np.rint(.55*rgb[mask] + .45*np.array([35, 220, 120])).astype(np.uint8)
            Image.fromarray(rgb).save(paths['rgb'])
            Image.fromarray(mask.astype(np.uint8)*255).save(paths['mask'])
            Image.fromarray(overlay).save(paths['overlay'])
            np.save(paths['logits'], logits, allow_pickle=False)
            coverage = []
            for box in rois.get(Path(row['path']).name, []):
                x0, y0, x1, y1 = box
                region = mask[y0:y1, x0:x1]
                if region.shape != (y1-y0, x1-x0):
                    raise ValueError('Original annotation outside source grid')
                coverage.append(dict(xyxy=box, pixels=int(region.size),
                                     selected=int(region.sum()), fraction=float(region.mean())))
            record = dict(index=index, source_sha256=row['sha256'], source_path=row['path'], role='train',
                          shape=list(cube.shape), positive_foreground=int(foreground.sum()),
                          qualified_pixels=int(mask.sum()), sampled=n,
                          failed_minimum=n == 0, sampled_above_one_channels=int((spectra > 1).sum()),
                          sampled_minimum=float(spectra.min()) if n else None,
                          sampled_maximum=float(spectra.max()) if n else None,
                          sampled_srgb_out_of_gamut_channels=int(((encoded[y, xcoord] < 0) |
                                                                 (encoded[y, xcoord] > 1)).sum()),
                          mask_is_ground_truth=False, positive_only_roi_coverage=coverage,
                          previews={k: str(p) for k, p in paths.items()},
                          preview_sha256={k: digest(p) for k, p in paths.items()})
            records.append(record)
            tiles.append(preview_tile(rgb, overlay, f'TRAIN {index+1:02d} | {row["sha256"][:8]} | derived D65'))
            print('PALETTE', index+1, '/ 19', 'qualified', int(mask.sum()), 'samples', n, flush=True)
            del cube, encoded, rgb, overlay, mask, foreground
    arrays = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    for suffix, matrix, white in zip(('2deg', '10deg'), matrices, whites, strict=True):
        arrays['matrix_'+suffix] = matrix
        arrays['white_'+suffix] = white
        arrays['xyz_d65_'+suffix] = arrays['spectra'] @ matrix
        arrays['lab_d65_'+suffix] = xyz_lab(arrays['xyz_d65_'+suffix], white)
    arrays.update(wavelength_nm=knots, integration_wave_nm=wave, cmf_2deg=cmf2, cmf_10deg=cmf10, d65_spd=spd,
                  srgb_d65_unclipped=linear_to_srgb(arrays['xyz_d65_2deg'] @ inverse_rgb),
                  source_sha256=np.array([r['source_sha256'] for r in records]),
                  source_role=np.array(['train']*19))
    np.savez_compressed(OUT / 'palette.npz', **arrays)
    sheet = Image.new('RGB', (544*4, 348*5), (22, 26, 33))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 4)*544, (i//4)*348))
    sheet.save(OUT / 'contact_sheet.png')
    for start in range(0, len(tiles), 4):
        page = Image.new('RGB', (1088, 696), (22, 26, 33))
        for j, tile in enumerate(tiles[start:start+4]):
            page.paste(tile, ((j % 2)*544, (j//2)*348))
        page.save(OUT / f'review_page_{start//4+1:02d}.png')
    if torch.cuda.is_initialized():
        raise ValueError('Unexpected CUDA initialization')
    verify_lock()
    report = dict(protocol_sha256=digest(OUT / 'protocol.json'), palette_sha256=digest(OUT / 'palette.npz'),
                  palette_bytes=(OUT / 'palette.npz').stat().st_size,
                  source_groups=19, successful_source_groups=sum(r['sampled'] > 0 for r in records),
                  samples=len(arrays['spectra']), qualified_pixels=sum(r['qualified_pixels'] for r in records),
                  sampled_above_one_channels=int((arrays['spectra'] > 1).sum()),
                  weight_convention='Each nonempty source sums to 1; total equals nonempty source count',
                  white_xyz_y1={k: w.tolist() for k, w in zip(('2deg', '10deg'), whites, strict=True)},
                  outside400_720_xyz_weight_mass={k: m.tolist() for k, m in zip(('2deg', '10deg'), missing, strict=True)},
                  observer10_zbar_zero_extrapolation=True, reflectance_tail_assumption='constant endpoints',
                  reflectance_clipped=False, sampled_colors_clipped=False, cuda_initialized=False,
                  held_sources_decoded=0, new_camera_photographs=0, native_phone_color_targets=0,
                  automated_mask_not_ground_truth=True, visual_review_pending=True,
                  derived_renderings_not_independent_camera_evidence=True, files=records,
                  seconds=time.perf_counter()-started,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__))
    save_json(OUT / 'profile.json', report)
    print('PALETTE COMPLETE', json.dumps({k: v for k, v in report.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'prepare', 'verify'])
    action = parser.parse_args().action
    if action == 'freeze':
        freeze()
    elif action == 'prepare':
        prepare()
    else:
        verify_lock()
        print('PALETTE INPUT LOCK VERIFIED')
