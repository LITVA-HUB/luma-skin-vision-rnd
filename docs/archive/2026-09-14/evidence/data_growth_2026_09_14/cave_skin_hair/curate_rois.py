"""Small visually inspected material ROIs; keep the printed face out of live-skin data."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent / 'prepared_raw_v1'
OUT = ROOT / 'material_rois_v1'
OUT.mkdir(exist_ok=False)
ROIS = [
    ('face', 'visible_skin', (280, 165, 352, 197)),
    ('face', 'visible_skin', (225, 286, 266, 328)),
    ('face', 'visible_skin', (367, 285, 401, 324)),
    ('photo_and_face', 'visible_skin', (383, 240, 420, 266)),
    ('photo_and_face', 'visible_skin', (347, 314, 371, 338)),
    ('photo_and_face', 'visible_skin', (433, 311, 455, 335)),
    ('photo_and_face', 'printed_face', (85, 280, 123, 306)),
    ('photo_and_face', 'printed_face', (54, 339, 76, 365)),
    ('photo_and_face', 'printed_face', (139, 335, 160, 361)),
    ('hairs', 'hair_appearance', (65, 180, 145, 340)),
    ('hairs', 'hair_appearance', (224, 175, 288, 335)),
    ('hairs', 'hair_appearance', (360, 165, 408, 325)),
]
source = json.loads((ROOT / 'profile.json').read_text())
arrays, source_bindings = {}, {}
for record in source['records']:
    path = Path(record['prepared_path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['prepared_sha256']
    with np.load(path, allow_pickle=False) as a:
        arrays[record['scene']] = {k:a[k].copy() for k in a.files}
    source_bindings[str(path)] = record['prepared_sha256']
rows, spectral, rgb = [], [], []
for roi_index, (scene, category, (left, top, right, bottom)) in enumerate(ROIS):
    assert 0 <= left < right <= 512 and 0 <= top < bottom <= 512
    positions = [(x, y) for y in range(top, bottom-15, 16) for x in range(left, right-15, 16)]
    positions.sort(key=lambda xy: hashlib.sha256(f'CAVE_ROI_V1|{roi_index}|{xy}'.encode()).digest())
    for x, y in positions[:8]:
        rows.append(dict(roi=roi_index, scene=scene, category=category, x=x, y=y, size=16,
                         group='cave_skin_and_hair_entire_acquisition'))
        spectral.append(arrays[scene]['bands_uint16'][y:y+16, x:x+16])
        rgb.append(arrays[scene]['author_rendered_srgb'][y:y+16, x:x+16])
destination = OUT / 'patches_uint16.npz'
np.savez_compressed(destination, bands_uint16=np.stack(spectral), author_rendered_srgb=np.stack(rgb),
                    wavelength_nm=np.arange(400, 701, 10, dtype=np.int16),
                    category=np.array([r['category'] for r in rows]),
                    scene=np.array([r['scene'] for r in rows]))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = dict(visible_skin='#54eb97', printed_face='#f5b347', hair_appearance='#72aaff')
for ax, scene in zip(axes, ['face', 'photo_and_face', 'hairs'], strict=True):
    ax.imshow(arrays[scene]['author_rendered_srgb'])
    for i, (s, category, (x1, y1, x2, y2)) in enumerate(ROIS):
        if s == scene:
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=colors[category], lw=1.5))
            ax.text(x1, y1-4, str(i), color=colors[category], fontsize=9)
    ax.set_title(scene); ax.axis('off')
fig.suptitle('CAVE: selected material regions (green: skin; orange: print; blue: hair appearance)')
fig.tight_layout(); fig.savefig(OUT/'roi_review.png',dpi=130);plt.close(fig)
manifest = dict(source_profile_sha256=hashlib.sha256((ROOT/'profile.json').read_bytes()).hexdigest(),
                source_bindings=source_bindings, rois=[dict(scene=s, category=c, xyxy=list(b)) for s,c,b in ROIS],
                patches=rows, patch_counts=dict(Counter(r['category'] for r in rows)),
                sampling='Non-overlapping 16x16 grid within each rectangle; hash-selected at most 8 per rectangle.',
                annotation='Conservative rectangular visual material annotation; not a full skin segmentation mask.',
                review='Source BMPs visually inspected; printed face on left is excluded from visible_skin.',
                limitations=['All patches share one conservative acquisition group; no independent test claim.',
                             'Source rendered RGB is derived, not a separate camera capture.',
                             'No colorimetric target or reflectance scale conversion was generated.'],
                prepared_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                entered_registered_HR_or_P3=False)
with (OUT/'manifest.json').open('x',encoding='utf-8') as stream:
    json.dump(manifest,stream,indent=2);stream.write('\n')
print(json.dumps({'patch_counts':manifest['patch_counts'],'total':len(rows),
                  'prepared_sha256':manifest['prepared_sha256']}))
