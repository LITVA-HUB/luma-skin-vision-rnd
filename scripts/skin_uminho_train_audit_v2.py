"""Numerical inventory of the 19 acquired TRAIN cubes, without RGB rendering."""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from skin_uminho_train_expand_v2 import OUT, digest, save_once


def main():
    profile = json.loads((OUT / 'profile.json').read_text(encoding='utf-8'))
    if profile['total_cubes'] != 19 or profile['held_cubes_acquired']:
        raise ValueError('Expected the frozen TRAIN-only extension')
    records = []
    for i, row in enumerate(profile['files']):
        path = Path(row['path'])
        if digest(path) != row['sha256']:
            raise ValueError('Source cube checksum changed')
        data = loadmat(path, variable_names=['datao'], verify_compressed_data_integrity=True)['datao']
        if data.ndim != 3 or data.shape[-1] != 33 or data.dtype != np.float64:
            raise ValueError('Expected original float64 reflectance cube with 33 bands')
        finite = bool(np.isfinite(data).all())
        if not finite:
            raise ValueError('Non-finite source reflectance; do not silently repair')
        foreground = np.any(data != 0, axis=-1)
        values = data[foreground]
        record = dict(source_sha256=row['sha256'], role='train', shape=list(data.shape), dtype=str(data.dtype),
                      pixels=int(foreground.size), foreground_pixels=int(foreground.sum()),
                      artificial_zero_background_pixels=int((~foreground).sum()), finite=finite,
                      foreground_minimum=float(values.min()), foreground_maximum=float(values.max()),
                      negative_channels=int((values < 0).sum()), above_one_channels=int((values > 1).sum()),
                      channels=int(values.size), clipped=False, verified_skin_mask=False)
        records.append(record)
        print('UMINHO AUDIT', i+1, 'of', len(profile['files']), 'foreground spectra', record['foreground_pixels'], flush=True)
        del values, foreground, data
    result = dict(profile_sha256=digest(OUT / 'profile.json'), source_sha256=digest(__file__),
                  wavelength_nm=list(range(400, 721, 10)), cubes=len(records),
                  foreground_spectra=sum(r['foreground_pixels'] for r in records),
                  negative_channels=sum(r['negative_channels'] for r in records),
                  above_one_channels=sum(r['above_one_channels'] for r in records),
                  foreground_minimum=min(r['foreground_minimum'] for r in records),
                  foreground_maximum=max(r['foreground_maximum'] for r in records),
                  files=records, held_data_decoded=False, new_camera_photographs=0,
                  limitations=['Spectral pixels within a face are correlated, not independent participants',
                               'Nonzero foreground includes facial features and is not a skin-only annotation',
                               'Values outside 0..1 preserved; directional reflectance/artifacts require review',
                               'Finite 400..720nm band range; rendered RGB is not independent camera evidence'])
    save_once(OUT / 'numeric_audit.json', result)
    print('UMINHO AUDIT COMPLETE', json.dumps({k: v for k, v in result.items() if k not in ['files', 'wavelength_nm', 'limitations']}), flush=True)


if __name__ == '__main__':
    main()
