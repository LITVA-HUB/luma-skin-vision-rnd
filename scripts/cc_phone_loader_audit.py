"""Development-only audit of frozen Beyond RGB TRAIN scenes; no test scoring.

Read already demosaiced camera RGB as released. Never apply a second Bayer
conversion, black subtraction, white-level division, white balance or CCM.
Patch statistics diagnose reference quality; they are not model accuracy.
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT/'docs/data/provenance/mobile_screen_2026_09_11'
LOCK = PROVENANCE/'beyond_rgb_selection.json'


def loader_scenes(lock, requested=None):
    allowed = [s for s in lock['scenes'] if s['role'] == 'loader']
    if requested is not None:
        if set(requested) - {s['scene'] for s in allowed}:
            raise ValueError('Requested reserved or unknown scene; audit permits loader roles only')
        allowed = [s for s in allowed if s['scene'] in requested]
    if not allowed:
        raise ValueError('No loader scenes')
    return allowed


@contextmanager
def open_camera_rgb(path, camera):
    with h5py.File(path, 'r') as f:
        if list(f.keys()) != [camera]:
            raise ValueError('Unexpected camera HDF5 key')
        data = f[camera]
        if data.ndim != 3 or data.shape[-1] != 3 or data.dtype != np.dtype('float32'):
            raise ValueError('Expected released HWC float32 demosaiced camera RGB')
        yield data


def radiometric_sample(data, stride=8):
    if stride < 1:
        raise ValueError('Positive stride required')
    sample = np.asarray(data[::stride, ::stride, :], dtype=np.float64)
    if not np.isfinite(sample).all():
        raise ValueError('Image sample must be finite')
    return {
        'stride': stride, 'shape': list(data.shape), 'sampled_values': sample.size,
        'min': float(sample.min()), 'max': float(sample.max()),
        'quantiles': np.quantile(sample, [0, .01, .5, .99, 1]).tolist(),
        'median_rgb': np.median(sample, axis=(0, 1)).tolist(),
        'fraction_on_8bit_grid': float(np.mean(abs(sample*255-np.round(sample*255)) < 2e-5)),
        'fraction_pixels_saturated': float(np.mean((sample >= 254/255).any(-1))),
        'fraction_pixels_nonpositive': float(np.mean((sample <= 0).any(-1))),
    }


def patch_stats(data, corners, inset=.5):
    """Sample pixel centres in the central convex quad, using stored (x,y)."""
    p = np.asarray(corners, dtype=np.float64)
    if p.shape != (4, 2) or not np.isfinite(p).all() or not 0 < inset <= 1:
        raise ValueError('Expected finite four-corner polygon and inset in (0,1]')
    if (p < 0).any() or (p[:, 0] >= data.shape[1]).any() or (p[:, 1] >= data.shape[0]).any():
        raise ValueError('Patch outside image bounds')
    edges = np.roll(p, -1, axis=0)-p
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0]*following[:, 1]-edges[:, 1]*following[:, 0]
    if not (np.all(turns > 0) or np.all(turns < 0)):
        raise ValueError('Patch corners must form ordered strictly convex quadrilateral')
    centre = p.mean(0)
    p = centre+inset*(p-centre)
    x0, y0 = np.floor(p.min(0)).astype(int)
    x1, y1 = np.ceil(p.max(0)).astype(int)
    yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
    inside = np.ones(xx.shape, dtype=bool)
    orientation = np.sign(turns[0])
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        cross = (b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0])
        inside &= orientation*cross >= -1e-9
    pixels = np.asarray(data[y0:y1+1, x0:x1+1, :], dtype=np.float64)[inside]
    if len(pixels) < 16 or not np.isfinite(pixels).all():
        raise ValueError('Patch has too few finite samples')
    median = np.median(pixels, axis=0)
    unit = median/max(np.linalg.norm(median), 1e-12)
    return {
        'median_rgb': median.tolist(), 'unit_rgb': unit.tolist(), 'sample_count': len(pixels),
        'saturated_pixel_fraction': float(np.mean((pixels >= 254/255).any(-1))),
        'nonpositive_pixel_fraction': float(np.mean((pixels <= 0).any(-1))),
        'rgb_p10': np.quantile(pixels, .1, axis=0).tolist(),
        'rgb_p90': np.quantile(pixels, .9, axis=0).tolist(),
        'inset_fraction': inset, 'corners_xy': np.asarray(corners).tolist(),
    }


def audit(args):
    if sha256(LOCK) != args.lock_sha256:
        raise ValueError('Frozen selection digest mismatch')
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    scenes = loader_scenes(lock)
    out = Path(args.out)
    if out.exists():
        raise ValueError('Use a new output directory; do not overwrite an audit')
    root = Path(args.data_root).resolve()
    records, files, panels = [], [], []
    for scene in scenes:
        base = root/'beyond-unzip/beyondRGB'/scene['scene']
        for camera in ('samsung', 'oppo'):
            row = {'scene': scene['scene'], 'role': 'loader', 'camera': camera}
            for role in ('NT', 'WT'):
                path = base/role/f'{camera}.h5'
                tags_path = base/role/f'{camera}_tags.json'
                tags = json.loads(tags_path.read_text(encoding='utf-8'))
                files.extend([{'path': str(p.relative_to(root)), 'sha256': sha256(p)} for p in (path, tags_path)])
                with open_camera_rgb(path, camera) as data:
                    row[role] = radiometric_sample(data)
                    row[role]['metadata'] = {k: tags.get(k) for k in (
                        'Model', 'Orientation', 'WhiteLevel', 'BlackLevel', 'CFAPattern',
                        'ColorMatrix1', 'ColorMatrix2', 'CameraCalibration1', 'CameraCalibration2')}
                    if role == 'WT':
                        detection_path = base/role/f'{camera}_cc_detection.json'
                        detection = json.loads(detection_path.read_text(encoding='utf-8'))
                        files.append({'path': str(detection_path.relative_to(root)), 'sha256': sha256(detection_path)})
                        patches = {str(i): patch_stats(data, detection[f'patch_{i}']['corners']) for i in range(1, 25)}
                        row['patches'] = patches
                        units = np.array([patches[str(i)]['unit_rgb'] for i in range(20, 24)])
                        cross = np.linalg.norm(np.cross(units[:, None], units[None, :]), axis=-1)
                        pairwise = np.degrees(np.arctan2(cross, units @ units.T))
                        row['neutral_20_23_max_pairwise_degrees'] = float(pairwise.max())
                        panels.append((scene['scene'], camera, np.array(data[::8, ::8]), detection))
            records.append(row)
    out.mkdir(parents=True)
    write_json(out/'audit.json', {
        'status': 'LOADER TRAIN SCENES ONLY; REFERENCE DIAGNOSTICS, NOT BENCHMARK RESULTS',
        'lock_sha256': args.lock_sha256, 'script_sha256': sha256(__file__),
        'h5py_version': h5py.__version__, 'numpy_version': np.__version__,
        'transforms': 'None. Released HWC camera RGB; no further Bayer, black/white, CCM, gamma or AWB.',
        'radiometry_scope': 'Strided sample, not exhaustive pixel validation.',
        'patches_scope': 'Full pixels within central 50% polygons. Patch20-23 disagreement diagnostic only.',
        'input_files': files, 'records': records})
    # Local development visualisation only; never commit scene images.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))
    for ax, (scene, camera, image, detection) in zip(axes.flat, panels):
        ax.imshow(np.clip(image, 0, 1)**(1/2.2))
        for i in range(19, 25):
            p = np.array(detection[f'patch_{i}']['corners'])/8
            ax.plot(*np.vstack([p, p[:1]]).T, color='magenta', linewidth=.8)
            ax.text(*p.mean(0), str(i), fontsize=8, color='yellow')
        ax.set_title(f'{scene} {camera}: WT, display gamma only')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out/'loader_reference_preview.png', dpi=130)
    plt.close(fig)
    print(json.dumps({'out': str(out), 'records': len(records), 'reserved_test_decoded': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--data-root', default=str(ROOT/'data/public/beyond_rgb_phone'))
    parser.add_argument('--out', required=True)
    audit(parser.parse_args())
