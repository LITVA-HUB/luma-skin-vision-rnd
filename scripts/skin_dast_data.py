"""Preserve the public DAST sample and its native, site-specific measurements."""
from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image
from skin_data_growth import DATA_ROOT, sha_bytes, write_json


def parse_cmf(text):
    rows = list(csv.reader(io.StringIO(text), delimiter='\t'))
    start = next((i for i, row in enumerate(rows) if row and row[0].strip() == 'Label'), None)
    if start is None:
        raise ValueError('missing colorimeter header')
    header = [value.strip() for value in rows[start]]
    keys = ['Label', 'CIE-L*', 'CIE-a*', 'CIE-b*', 'X', 'Y', 'Z']
    if not set(keys) <= set(header):
        raise ValueError('incomplete colorimeter header')
    result = []
    for row in rows[start+1:]:
        if not row or not row[0].strip():
            continue
        item = dict(zip(header, row, strict=False))
        values = [float(item[key]) for key in keys[1:]]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('colorimeter values must be finite')
        # Keep the author's codes. Sample 036 has "Measure 1...5" without site names.
        result.append(dict(site_code=item['Label'].strip(), anatomical_site=None,
                           lab_native=values[:3], xyz_native=values[3:],
                           illuminant=None, observer=None,
                           reference_status='native instrument values; convention and site mapping require confirmation'))
    if not result or len({r['site_code'] for r in result}) != len(result):
        raise ValueError('empty or duplicate colorimeter sites')
    return result


def ingest_dast(archive, output):
    archive, output = Path(archive), Path(output).resolve()
    with zipfile.ZipFile(archive) as z:
        members = [item for item in z.infolist() if not item.is_dir()]
        names = [item.filename for item in members]
        if len(set(names)) != len(names):
            raise ValueError('duplicate archive paths')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or p.parts[0] != 'example-data':
                raise ValueError('unsafe archive path')
            target = (output / Path(*p.parts[1:])).resolve()
            if not target.is_relative_to(output):
                raise ValueError('archive path escapes output')
        if sum(item.file_size for item in members) > 100_000_000:
            raise ValueError('unexpected expanded archive size')
        allowed = [name for name in names if re.fullmatch(
            r'example-data/(?:example_overview\.csv|\d{3}/Colorimeter/\d{3}\.cmf|\d{3}/Pictures/cropped/[\w-]+\.jpg)', name)]
        unknown = set(names) - set(allowed)
        if any(not name.endswith('/Thumbs.db') for name in unknown):
            raise ValueError('unexpected archive file')
        payloads = {name:z.read(name) for name in allowed}
    overview = payloads['example-data/example_overview.csv'].decode('utf-8-sig')
    metadata = list(csv.reader(io.StringIO(overview), delimiter=';'))
    measurements, images = {}, []
    for name, raw in payloads.items():
        if name.endswith('.cmf'):
            subject = PurePosixPath(name).parts[1]
            measurements[subject] = parse_cmf(raw.decode('utf-8-sig'))
    mapped = set()
    for row in metadata:
        if len(row) != 17 or not re.fullmatch(r'\d+', row[0]) or not re.fullmatch(r'I\d+', row[1]):
            raise ValueError('unexpected overview row')
        subject = f'{int(row[0]):03d}'
        expected = f'example-data/{subject}/Pictures/cropped/{subject}-{row[1]}.jpg'
        alias = expected.removesuffix('.jpg') + '-0.jpg'
        matches = [name for name in (expected, alias) if name in payloads]
        if len(matches) != 1 or matches[0] in mapped:
            raise ValueError('ambiguous or missing photograph')
        name = matches[0]
        if row[3] != expected.replace('example-data/', 'subjects/', 1):
            raise ValueError('overview path disagrees with subject/image key')
        mapped.add(name)
        if subject not in measurements:
            raise ValueError('missing subject colorimeter table')
        raw = payloads[name]
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            size, mode = list(im.size), im.mode
        images.append(dict(subject_id=subject, image_id=row[1], camera=row[8] or None,
                           exposure=row[10] or None, author_exposure_code=row[9] or None,
                           path=str(output / Path(*PurePosixPath(name).parts[1:])),
                           sha256=sha_bytes(raw), bytes=len(raw), size=size, mode=mode,
                           filename_alias_used=name != expected, role='external_diagnostic',
                           ground_truth_lab=None, measurement_table_subject=subject,
                           note='site-specific instrument table exists; no invented one-color whole-face target'))
    if mapped != {name for name in payloads if name.endswith('.jpg')}:
        raise ValueError('unmapped photographs')
    # All validation precedes writes. Original archive and every extracted payload remain byte exact.
    for name, raw in payloads.items():
        target = output / Path(*PurePosixPath(name).parts[1:])
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('existing original differs')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return dict(photographs=len(images), subjects=len(measurements),
                measurement_count=sum(map(len, measurements.values())),
                cameras=dict(Counter(row['camera'] or 'missing' for row in images)),
                exposures=dict(Counter(row['exposure'] or 'missing' for row in images)),
                images=images, measurements=measurements, skipped_files=sorted(unknown),
                archive_sha256=sha_bytes(archive.read_bytes()),
                public_example_only=True, participant_grouping='author subject directories',
                license_status='no explicit license found in original public repository; local research per user instruction',
                commercial_use_cleared=False,
                citation='Busch, Kibler, Stockhardt, Rathgeb (2026). A Skin Tone Annotated Face Image Dataset for Studying Demographic Variability. IWBF.',
                source_url='https://github.com/dasec/DAST-SkinTone-database',
                code_sha256=sha_bytes(Path(__file__).read_bytes()))


def main():
    root = DATA_ROOT / 'dast_public_example'
    receipt = json.loads((root/'example-data.zip.source.json').read_text(encoding='utf-8'))
    if sha_bytes((root/'example-data.zip').read_bytes()) != receipt['sha256']:
        raise ValueError('original archive checksum changed')
    result = ingest_dast(root/'example-data.zip', root/'originals')
    write_json(root/'profile.json', result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('images','measurements')}))


if __name__ == '__main__':
    main()
