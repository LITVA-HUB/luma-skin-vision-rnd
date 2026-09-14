"""Validate original CAVE spectral planes without creating measured camera targets."""
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'prepared_raw_v1'
OUT.mkdir(exist_ok=False)
records = []
for archive in sorted(ROOT.glob('*_ms.zip')):
    receipt = json.loads(archive.with_suffix('.zip.source.json').read_text())
    raw_archive = archive.read_bytes()
    assert hashlib.sha256(raw_archive).hexdigest() == receipt['sha256']
    scene = archive.stem.removesuffix('_ms')
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        assert z.testzip() is None
        bands = {}
        for name in z.namelist():
            match = re.fullmatch(re.escape(scene)+'_ms/(?:'+re.escape(scene)+r')_ms_(\d{2})\.png', name)
            if match:
                raw = z.read(name)
                with Image.open(io.BytesIO(raw)) as im:
                    a = np.array(im)
                    assert im.format == 'PNG' and a.shape == (512, 512) and a.dtype == np.uint16
                bands[int(match[1])] = (a, hashlib.sha256(raw).hexdigest())
        assert set(bands) == set(range(1, 32))
        cube = np.stack([bands[i][0] for i in range(1, 32)], axis=-1)
        preview_name = scene+'_ms/'+scene+'_RGB.bmp'
        preview_raw = z.read(preview_name)
        with Image.open(io.BytesIO(preview_raw)) as im:
            im.load()
            assert im.size == (512, 512) and im.mode == 'RGB'
            preview = np.asarray(im).copy()
        (OUT/(scene+'_original_RGB.bmp')).write_bytes(preview_raw)
        destination = OUT/(scene+'_uint16.npz')
        np.savez_compressed(destination, bands_uint16=cube,
                            wavelength_nm=np.arange(400, 701, 10, dtype=np.int16),
                            author_rendered_srgb=preview)
        with np.load(destination, allow_pickle=False) as check:
            assert np.array_equal(check['bands_uint16'], cube)
            assert np.array_equal(check['author_rendered_srgb'], preview)
        record = dict(scene=scene, source_url=receipt['url'], source_sha256=receipt['sha256'],
                      source_bytes=receipt['bytes'], shape=list(cube.shape), dtype=str(cube.dtype),
                      min_uint16=int(cube.min()), max_uint16=int(cube.max()),
                      zero_values=int((cube == 0).sum()), max_code_values=int((cube == 65535).sum()),
                      spectral_png_sha256={str(i):bands[i][1] for i in range(1, 32)},
                      author_rendered_rgb_sha256=hashlib.sha256(preview_raw).hexdigest(),
                      prepared_path=str(destination), prepared_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                      all_archive_member_crcs_checked=True,
                      source_physical_scope='Author-calibrated approximate reflectance, not exact instrument truth.',
                      scaling='Original uint16 codes retained. No scale conversion or Lab target generated.',
                      rgb_kind='Author-rendered sRGB under D65 from spectra, not an independent camera photograph.',
                      skin_mask=None, person_id=None, role='auxiliary_pool_unassigned',
                      conservative_group='cave_skin_and_hair_entire_acquisition')
        records.append(record)
profile = dict(source='https://cave.cs.columbia.edu/repository/Multispectral', scenes=len(records),
               independent_camera_photographs=0, manually_labeled_skin_masks=0,
               spectral_planes=sum(r['shape'][2] for r in records),
               spatial_spectra=sum(r['shape'][0]*r['shape'][1] for r in records),
               spatial_spectra_are_not_independent_examples=True,
               records=records, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               entered_registered_HR_or_P3=False)
with (OUT/'profile.json').open('x',encoding='utf-8') as stream:
    json.dump(profile,stream,indent=2);stream.write('\n')
print(json.dumps({k:v for k,v in profile.items() if k!='records'}))
for r in records:
    print(r['scene'],r['shape'],r['min_uint16'],r['max_uint16'])
