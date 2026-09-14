"""Read back every palette sample and independently integrate its original spectrum."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/Luma-RnD/data_growth_2026_09_14/uminho_palette_v1')
LEGACY = Path('C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/public/uminho_hsfd_v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def reference_integration(spectra, knots, wave, cmf, illuminant):
    """Interpolate each measured spectrum directly, without production matrices."""
    normalizer = math.fsum((cmf[:, 1]*illuminant).tolist())
    output = np.empty((len(spectra), 3), np.float64)
    weights = cmf*illuminant[:, None]
    for i, spectrum in enumerate(spectra):
        curve = np.interp(wave, knots, spectrum)
        for c in range(3):
            output[i, c] = math.fsum((curve*weights[:, c]).tolist())/normalizer
    white = np.array([math.fsum(weights[:, c].tolist())/normalizer for c in range(3)])
    ratio = output/white
    f = np.empty_like(ratio)
    nonlinear = ratio > (6/29)**3
    f[nonlinear] = ratio[nonlinear]**(1/3)
    f[~nonlinear] = ratio[~nonlinear]*(29/6)**2/3 + 4/29
    lab = np.column_stack((116*f[:, 1]-16, 500*(f[:, 0]-f[:, 1]), 200*(f[:, 1]-f[:, 2])))
    return output, lab, white


def erode_twice(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    for _ in range(2):
        padded = np.pad(mask, 1, constant_values=False)
        mask = np.logical_and.reduce([padded[y:y+h, x:x+w] for y in range(3) for x in range(3)])
    return mask


def main():
    import torch
    from PIL import Image
    from scipy.io import loadmat
    from skin_face_segment import SkinUNet
    from threadpoolctl import threadpool_limits

    sys.path.insert(0, str(ROOT / 'src'))
    from luma_skin_vision.color import RGB_XYZ, linear_to_srgb

    started = time.perf_counter()
    primary = read(OUT / 'protocol.json')
    profile = read(OUT / 'profile.json')
    contract = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sha(__file__), primary_protocol_sha256=sha(OUT / 'protocol.json'),
                    palette_sha256=sha(OUT / 'palette.npz'), profile_sha256=sha(OUT / 'profile.json'),
                    xyz_atol=1e-10, lab_atol=1e-8, srgb_atol=1e-10, logit_atol=0., rtol=0.,
                    scope='All original coordinates, deterministic masks/sampling, direct spectral integration; CPU only')
    if (OUT / 'audit.json').exists():
        raise ValueError('Preserve completed audit')
    save(OUT / 'audit_protocol.json', contract)
    if (contract['primary_protocol_sha256'] != profile['protocol_sha256']
            or contract['palette_sha256'] != profile['palette_sha256']):
        raise ValueError('Primary output does not match its receipt')
    for path, expected in primary['bindings'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    manifest = read(LEGACY / 'manifest.json')['rows']
    training_names = sorted(r['name'] for r in manifest if r['role'] == 'train')
    if training_names != [Path(r['path']).name for r in primary['sources']]:
        raise ValueError('A source is missing, duplicated, held, or reordered')
    z = dict(np.load(OUT / 'palette.npz', allow_pickle=False))
    if (z['spectra'].dtype != np.float64 or z['spectra'].shape != (19456, 33)
            or len(np.unique(z['source_sha256'])) != 19 or not (z['source_role'] == 'train').all()
            or not (z['spectra'] > 0).all() or not np.isfinite(z['spectra']).all()):
        raise ValueError('Unexpected measured palette shape/type/support')
    np.testing.assert_array_equal(z['source_sha256'], [r['sha256'] for r in primary['sources']])
    np.testing.assert_array_equal(z['wavelength_nm'], np.arange(400, 721, 10))
    np.testing.assert_array_equal(z['integration_wave_nm'], np.arange(360, 831))
    # Re-read the original tables rather than trusting the cached integration data.
    cie2_path = next(Path(p) for p in primary['bindings'] if p.endswith('CIE_xyz_1931_2deg.csv'))
    cie10_path = next(Path(p) for p in primary['bindings'] if p.endswith('CIE_xyz_1964_10deg.csv'))
    d65_path = next(Path(p) for p in primary['bindings'] if p.endswith('CIE_std_illum_D65.csv'))
    cie2, cie10, d65 = [np.genfromtxt(p, delimiter=',') for p in (cie2_path, cie10_path, d65_path)]
    np.testing.assert_array_equal(np.isnan(cie10[:, 3]), cie10[:, 0] > 559)
    cie10[cie10[:, 0] > 559, 3] = 0
    spd = np.interp(z['integration_wave_nm'], d65[:, 0], d65[:, 1])
    np.testing.assert_array_equal(z['cmf_2deg'], cie2[:, 1:])
    np.testing.assert_array_equal(z['cmf_10deg'], cie10[:, 1:])
    np.testing.assert_array_equal(z['d65_spd'], spd)
    if torch.cuda.is_initialized():
        raise ValueError('CPU-only audit required')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = next(Path(p) for p in primary['bindings'] if p.endswith('/best.pt') or p.endswith('\\best.pt'))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = SkinUNet(width=state['width']).eval()
    model.load_state_dict(state['state_dict'], strict=True)
    roi_map = {r['name']: boxes for r, boxes in zip(read(LEGACY / 'train_expand_v1.json')['rows'],
               read(LEGACY / 'skin_regions_v1.json'), strict=True)}
    records = []
    max_xyz, max_lab, max_rgb, max_logit = 0., 0., 0., 0.
    with threadpool_limits(limits=1), torch.inference_mode():
        for i, (source, record) in enumerate(zip(primary['sources'], profile['files'], strict=True)):
            if (record['index'] != i or record['source_sha256'] != source['sha256']
                    or source['role'] != 'train' or sha(source['path']) != source['sha256']):
                raise ValueError('Source provenance mismatch')
            cube = loadmat(source['path'], variable_names=['datao'],
                           verify_compressed_data_integrity=True)['datao']
            rows = np.flatnonzero(z['source_index'] == i)
            coords = z['pixel_yx'][rows]
            if not ((coords >= 0).all() and (coords < np.array(cube.shape[:2])).all()):
                raise ValueError('Out-of-source coordinates')
            observed = cube[coords[:, 0], coords[:, 1]]
            np.testing.assert_array_equal(z['spectra'][rows], observed)
            np.testing.assert_array_equal(z['source_weight'][rows], np.full(len(rows), 1/len(rows)))
            if abs(z['source_weight'][rows].sum()-1) > 1e-12 or len(np.unique(coords, axis=0)) != len(rows):
                raise ValueError('Group weights or no-replacement sampling failed')
            paths = {k: Path(p) for k, p in record['previews'].items()}
            for key, path in paths.items():
                if sha(path) != record['preview_sha256'][key]:
                    raise ValueError('Changed preview or logit artifact')
            rgb = np.asarray(Image.open(paths['rgb']).convert('RGB'))
            rendered = linear_to_srgb((cube @ z['matrix_2deg']) @ np.linalg.inv(RGB_XYZ).T)
            np.testing.assert_array_equal(rgb, np.rint(np.clip(rendered, 0, 1)*255).astype(np.uint8))
            resized = np.asarray(Image.fromarray(rgb).resize((192, 192), Image.Resampling.BILINEAR))
            tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0).float()/255
            repeated_logits = model(tensor)[0, 0].numpy()
            stored_logits = np.load(paths['logits'], allow_pickle=False)
            max_logit = max(max_logit, float(np.max(np.abs(repeated_logits-stored_logits))))
            np.testing.assert_array_equal(repeated_logits, stored_logits)
            small = erode_twice(stored_logits >= math.log(19))
            mask = np.asarray(Image.fromarray(small.astype(np.uint8)*255).resize(
                (cube.shape[1], cube.shape[0]), Image.Resampling.NEAREST)) > 0
            mask &= np.all(cube > 0, axis=-1)
            np.testing.assert_array_equal(mask, np.asarray(Image.open(paths['mask'])) > 0)
            if int(mask.sum()) != record['qualified_pixels']:
                raise ValueError('Incorrect qualified-pixel count')
            pool = np.flatnonzero(mask)
            seed = int.from_bytes(hashlib.sha256(('LumaSpectralPaletteV1|'+source['sha256']).encode()).digest(), 'big')
            chosen = np.random.default_rng(seed).choice(pool, min(1024, len(pool)), replace=False)
            np.testing.assert_array_equal(np.ravel_multi_index(coords.T, mask.shape), chosen)
            if not mask[coords[:, 0], coords[:, 1]].all() or len(rows) != record['sampled']:
                raise ValueError('A selected spectrum is outside the frozen mask')
            for suffix, cmf in [('2deg', cie2[:, 1:]), ('10deg', cie10[:, 1:])]:
                xyz, lab, white = reference_integration(observed, z['wavelength_nm'],
                    z['integration_wave_nm'], cmf, spd)
                dx = float(np.max(np.abs(xyz-z['xyz_d65_'+suffix][rows])))
                dl = float(np.max(np.abs(lab-z['lab_d65_'+suffix][rows])))
                max_xyz, max_lab = max(max_xyz, dx), max(max_lab, dl)
                np.testing.assert_allclose(z['xyz_d65_'+suffix][rows], xyz, rtol=0, atol=contract['xyz_atol'])
                np.testing.assert_allclose(z['lab_d65_'+suffix][rows], lab, rtol=0, atol=contract['lab_atol'])
                np.testing.assert_allclose(z['white_'+suffix], white, rtol=0, atol=contract['xyz_atol'])
                if suffix == '2deg':
                    linear_rgb = np.linalg.solve(RGB_XYZ, xyz.T).T
                    encoded = np.empty_like(linear_rgb)
                    low = linear_rgb <= .0031308
                    encoded[low] = linear_rgb[low]*12.92
                    encoded[~low] = 1.055*linear_rgb[~low]**(1/2.4)-.055
                    max_rgb = max(max_rgb, float(np.max(np.abs(encoded-z['srgb_d65_unclipped'][rows]))))
                    np.testing.assert_allclose(z['srgb_d65_unclipped'][rows], encoded,
                                               rtol=0, atol=contract['srgb_atol'])
            boxes = roi_map.get(Path(source['path']).name, [])
            coverage = record['positive_only_roi_coverage']
            if len(boxes) != len(coverage):
                raise ValueError('Missing original positive-only annotation')
            for box, original in zip(boxes, coverage, strict=True):
                x0, y0, x1, y1 = box
                region = mask[y0:y1, x0:x1]
                if (original['xyxy'] != box or original['selected'] != int(region.sum())
                        or original['pixels'] != region.size or original['fraction'] != float(region.mean())):
                    raise ValueError('Positive-only annotation accounting mismatch')
            records.append(dict(source_index=i, samples=len(rows), original_coordinates_exact=True,
                                repeated_logits_exact=True, masks_exact=True, deterministic_sampling_exact=True))
            print('PALETTE AUDIT', i+1, '/ 19', len(rows), 'original spectra exact', flush=True)
            del cube, rendered, observed
    if torch.cuda.is_initialized() or len(records) != 19:
        raise ValueError('Invalid audit completion')
    if (sha(__file__) != contract['source_sha256'] or sha(OUT / 'profile.json') != contract['profile_sha256']
            or sha(OUT / 'palette.npz') != contract['palette_sha256']):
        raise ValueError('Audited artifact changed during execution')
    result = dict(audit_protocol_sha256=sha(OUT / 'audit_protocol.json'), status='passed',
                  sources_checked=len(records), samples_checked=len(z['spectra']),
                  observers_checked=2, max_xyz_drift=max_xyz, max_lab_drift=max_lab,
                  max_srgb_drift=max_rgb, max_logit_drift=max_logit,
                  source_weights_total=float(z['source_weight'].sum()),
                  above_one_channels_preserved=int((z['spectra'] > 1).sum()),
                  roi_regions_checked=sum(len(r['positive_only_roi_coverage']) for r in profile['files']),
                  held_source_names_metadata_only=True, held_cubes_decoded=0, cuda_initialized=False,
                  files=records, seconds=time.perf_counter()-started)
    save(OUT / 'audit.json', result)
    print('PALETTE AUDIT COMPLETE', json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
