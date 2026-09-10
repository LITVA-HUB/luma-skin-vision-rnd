"""Prepare only frozen phone TRAIN loader scenes under reference protocol v1."""
import argparse
import json
from pathlib import Path

import numpy as np
from cc_phone_loader_audit import LOCK, ROOT, loader_scenes, open_camera_rgb, patch_stats

from luma_skin_vision.cc.core import EXPERT_NAMES, angular, experts, reproduction, summarize
from luma_skin_vision.cc.data import sample
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

PROTOCOL = ROOT/'docs/research/phone_reference_protocol_v1.md'


def reference_from_patches(patches):
    candidates, diagnostics = [], {}
    for i in (20, 21, 22):
        p = patches[str(i)]
        rgb = np.asarray(p['median_rgb'], dtype=np.float64)
        okay = (rgb.shape == (3,) and np.isfinite(rgb).all() and np.min(rgb) >= 8/255
                and p['sample_count'] >= 64 and p['saturated_pixel_fraction'] <= .005)
        diagnostics[str(i)] = {'usable': bool(okay), **p}
        if okay:
            candidates.append((i, rgb/np.linalg.norm(rgb)))
    result = {'valid': False, 'gt': None, 'selected_patch': None,
              'usable_patches': [p[0] for p in candidates], 'diagnostics': diagnostics,
              'max_pairwise_degrees': None, 'reason': 'Fewer than two usable gray references'}
    if len(candidates) < 2:
        return result
    units = np.array([p[1] for p in candidates])
    cross = np.linalg.norm(np.cross(units[:, None], units[None, :]), axis=-1)
    max_angle = float(np.degrees(np.arctan2(cross, units @ units.T)).max())
    result['max_pairwise_degrees'] = max_angle
    if max_angle > 3:
        result['reason'] = 'Gray reference disagreement exceeds3degrees'
        return result
    result.update(valid=True, gt=candidates[0][1].tolist(), selected_patch=candidates[0][0], reason='Scorable chart reference')
    return result


def prepare(args):
    if sha256(LOCK) != args.lock_sha256 or sha256(PROTOCOL) != args.protocol_sha256:
        raise ValueError('Explicit selection/protocol hash mismatch')
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    scenes = loader_scenes(lock)  # This executable cannot prepare reserved_test roles.
    out = Path(args.out)
    if out.exists():
        raise ValueError('Existing output: use a new immutable run directory')
    root = Path(args.data_root).resolve()
    rows, images, targets, hypotheses, file_records = [], [], [], [], []
    for scene in scenes:
        base = root/'beyond-unzip/beyondRGB'/scene['scene']
        for camera in ('samsung', 'oppo'):
            nt_path, wt_path = [base/role/f'{camera}.h5' for role in ('NT', 'WT')]
            det_path = base/'WT'/f'{camera}_cc_detection.json'
            det = json.loads(det_path.read_text(encoding='utf-8'))
            with open_camera_rgb(wt_path, camera) as data:
                patches = {str(i): patch_stats(data, det[f'patch_{i}']['corners']) for i in range(1, 25)}
            ref = reference_from_patches(patches)
            row = {'id': f"beyond_rgb:{scene['scene']}:{camera}", 'scene': scene['scene'],
                   'camera': camera, 'split': 'loader', 'reference': ref,
                   'all_patch_diagnostics': patches, 'cache_index': None}
            for path in (nt_path, wt_path, det_path):
                file_records.append({'path': str(path.relative_to(root)), 'sha256': sha256(path)})
            if ref['valid']:
                with open_camera_rgb(nt_path, camera) as data:
                    rgb = np.array(data, dtype=np.float32)
                if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                    raise ValueError('Released RGB must be finite in[0,1]; do not silently rescale')
                mask = (rgb.max(-1) >= 254/255) | (rgb.max(-1) <= 0)
                row['masked_fraction'] = float(mask.mean())
                rgb[mask] = 0
                row['cache_index'] = len(images)
                hypotheses.append(experts(rgb))
                images.append(sample(rgb, size=128))
                targets.append(ref['gt'])
            rows.append(row)
            print(json.dumps({'id': row['id'], 'reference_valid': ref['valid']}), flush=True)
    if not images:
        raise ValueError('No scorable loader references')
    out.mkdir(parents=True)
    # Conventional cache fields match the existing public-CC model inputs.
    np.savez_compressed(out/'loader.npz', images=np.array(images), gt=np.array(targets),
                        experts=np.array(hypotheses))
    means = {}
    for k, name in enumerate(EXPERT_NAMES):
        pred = np.array(hypotheses)[:, k]
        means[name] = {'reproduction': summarize(reproduction(pred, np.array(targets))),
                       'recovery': summarize(angular(pred, np.array(targets)))}
    write_json(out/'loader_manifest.json', {
        'status': 'SIX LOADER TRAIN CAPTURES ONLY; NOT A HELD-OUT PHONE BENCHMARK',
        'lock_sha256': args.lock_sha256, 'protocol_sha256': args.protocol_sha256,
        'script_sha256': sha256(__file__), 'audit_script_sha256': sha256(ROOT/'scripts/cc_phone_loader_audit.py'),
        'preprocess_source_sha256': sha256(ROOT/'src/luma_skin_vision/cc/data.py'),
        'experts_source_sha256': sha256(ROOT/'src/luma_skin_vision/cc/core.py'),
        'planned_inputs': len(rows), 'scorable_inputs': len(images), 'reserved_test_decoded': 0,
        'input_files': file_records, 'cache_sha256': sha256(out/'loader.npz'),
        'rows': rows, 'classical_loader_diagnostics': means})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--data-root', default=str(ROOT/'data/public/beyond_rgb_phone'))
    parser.add_argument('--out', required=True)
    prepare(parser.parse_args())
