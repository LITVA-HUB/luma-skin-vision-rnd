"""Read-only qualification of DAST native color conventions, without inventing targets.

The numerical comparison recomputes Lab from the SAME stored XYZ with different
white references. It is neither an observer conversion nor a model-quality metric.
Original source files, existing data roles and model registrations are not edited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/Luma-RnD/data_growth_2026_09_14')
SOURCE = DATA / 'dast_public_example'
OUT = DATA / 'dast_qualification_v1'
PALETTE = DATA / 'uminho_palette_v1/profile.json'
PROFILE_SHA = '15db398c35c9c68427f3f09a2ff9f6636c9717c37953217ec71c6637807e8dd1'
PALETTE_SHA = '008034fd2c82648fa3aa316d37c35ce3a70320681bd40810c5b62472147f97f6'
DOCUMENTS = {
    'paper.pdf': 'ebdeafa82a5e55defbc28d5fbd92ef49f637b1abbe6f63bfda71313ff125d5a1',
    'slides.pdf': 'bf2bff1f38b0e4c1547cd9789036149aa9ed2ea337afaff739d99a45c0fe8732',
    'dsm4-manual-v8.pdf': 'ff9a34011c61850585b16336fd1bd66a6b6c607edef6713189dfbc81608f6786',
    'hunterlab-whitepoints.html': '9a101785a098400b6f3d72bf7c3adcaef8489b7ab46a6506ab235269594516d1',
    'dast-readme-current.md': '5ef81b1a83615e268202949fd4758f6aa5fe77c068c06fc14e83bdaa09272252',
}
CONVENTION = dict(
    illuminant='D65', observer_degrees=10, geometry='45/0',
    instrument='Cortex DSM-4', evidence='paper.pdf p4; dsm4-manual-v8.pdf pp6-7',
    qualification_basis='documented device convention plus consistency with native Lab/XYZ',
    manual_release='2026-02-10', native_software_version='3.2.0.7',
    limit='Manual is later than the 2025 measurements; the original session configuration is not independently certified.',
    native_to_srgb_conversion_performed=False,
)
PAPER_SITES = dict(CFH='forehead', CLC='left cheek, zygomatic region',
                   CRC='right cheek, zygomatic region', CLI='left hand, interosseous region',
                   CRI='right hand, interosseous region')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def triple(value):
    if len(value) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
        raise ValueError('expected three finite numbers')
    return list(value)


def f_lab(value):
    d = 6 / 29
    return value ** (1 / 3) if value > d**3 else value / (3*d*d) + 4/29


def f_inverse(value):
    d = 6 / 29
    return value**3 if value > d else 3*d*d*(value - 4/29)


def xyz_to_lab(xyz, white):
    xyz, white = triple(xyz), triple(white)
    if min(white) <= 0 or min(xyz) < 0:
        raise ValueError('expected positive white and nonnegative XYZ')
    x, y, z = [f_lab(v/w) for v, w in zip(xyz, white, strict=True)]
    return [116*y-16, 500*(x-y), 200*(y-z)]


def infer_white(lab, xyz):
    lightness, a, b = triple(lab)
    xyz = triple(xyz)
    fy = (lightness + 16) / 116
    factors = [f_inverse(fy + a/500), f_inverse(fy), f_inverse(fy - b/200)]
    if min(factors) <= 0 or min(xyz) <= 0:
        raise ValueError('white inference requires positive XYZ and Lab inverse factors')
    return [v/f for v, f in zip(xyz, factors, strict=True)]


def white_diagnostics(measurements, whites):
    if not measurements or set(whites) != {'2deg', '10deg'}:
        raise ValueError('expected measurements and both observer reference whites')
    candidates = {}
    for name, white in whites.items():
        residuals = [math.dist(xyz_to_lab(row['xyz_native'], white), triple(row['lab_native']))
                     for row in measurements]
        candidates[name] = dict(white_xyz_y100=white, mean_lab_residual=statistics.mean(residuals),
                                median_lab_residual=statistics.median(residuals),
                                max_lab_residual=max(residuals), per_measurement_residual=residuals)
    inferred = [infer_white(row['lab_native'], row['xyz_native']) for row in measurements]
    return dict(candidates=candidates,
                inferred_white_median=[statistics.median(row[i] for row in inferred) for i in range(3)],
                inferred_white_ranges=[[min(row[i] for row in inferred), max(row[i] for row in inferred)]
                                       for i in range(3)],
                interpretation='same-XYZ white-reference diagnostic; not observer conversion, physical color error, or model accuracy')


def qualify_profile(profile, whites):
    images, groups = profile['images'], profile['measurements']
    if not images or not groups or any(not rows for rows in groups.values()):
        raise ValueError('empty image or measurement group')
    if (profile['photographs'] != len(images) or profile['subjects'] != len(groups)
            or profile['measurement_count'] != sum(map(len, groups.values()))):
        raise ValueError('profile counts disagree with row grain')
    keys = [(row['subject_id'], row['image_id']) for row in images]
    if len(set(keys)) != len(keys) or {key[0] for key in keys} != set(groups):
        raise ValueError('duplicate image key or missing subject join')
    for row in images:
        if (row['role'] != 'external_diagnostic' or row['ground_truth_lab'] is not None
                or row['measurement_table_subject'] != row['subject_id']):
            raise ValueError('unexpected role, image target or subject association')
    native = []
    for subject, rows in groups.items():
        site_codes = [row['site_code'] for row in rows]
        if len(set(code.casefold() for code in site_codes)) != len(site_codes):
            raise ValueError('duplicate measurement site codes')
        for row in rows:
            triple(row['lab_native'])
            triple(row['xyz_native'])
            if row.get('anatomical_site') is not None:
                raise ValueError('v1 source unexpectedly already has a site crosswalk')
            native.append(dict(copy.deepcopy(row), subject_id=subject,
                               illuminant='D65', observer_degrees=10,
                               observer='CIE 1964 10-degree standard observer',
                               source_observer_annotation=row.get('observer'),
                               anatomical_site=None, paper_site_code=None,
                               convention_status='qualified_from_device_documentation_and_numeric_check',
                               reference_status='native site measurement; image-region link unresolved',
                               site_mapping_status='unresolved; do not infer from label abbreviation or row order',
                               approved_as_image_target=False))
    numeric = white_diagnostics(native, whites)
    # This is a data-consistency screen, not an acceptance threshold for a model.
    if numeric['candidates']['10deg']['max_lab_residual'] > .1:
        raise ValueError('native numbers do not support the documented D65/10 convention')
    return dict(
        counts=dict(photographs=len(images), author_subject_groups=len(groups),
                    native_measurements=len(native), qualified_conventions=len(native),
                    confirmed_site_mappings=0, eligible_camera_lab_pairs=0),
        images=copy.deepcopy(images), measurements=native, white_diagnostic=numeric,
        paper_site_definitions=PAPER_SITES,
        facial_class_method='Paper p4 averages the three facial sites CFH, CLC, CRC; hand sites are excluded.',
        face_mean_lab_computed=False,
        limitations=[
            'Paper CFH/CLC/CRC/CLI/CRI has no confirmed crosswalk to public S/LW/RW/LH/RH/Rh or Measure 1...5.',
            'No point coordinates or physical 4.2 mm measurement footprint is registered to any photograph.',
            'All photographs remain external diagnostics, including previously examined Seg1 examples.',
            'Three author-provided subject groups are not 36 independent people.',
            'Native 10-degree XYZ cannot become 2-degree camera/sRGB truth by changing its white point.',
        ],
    )


def audit_originals(root, profile):
    root = Path(root).resolve()
    archive_path = root / 'example-data.zip'
    if digest(archive_path) != profile['archive_sha256']:
        raise ValueError('original archive checksum changed')
    paths = [root / 'profile.json', archive_path]
    original_root = (root / 'originals').resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if len(set(names)) != len(names):
            raise ValueError('duplicate archive member')
        for name in names:
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name
                    or relative.parts[0] != 'example-data'):
                raise ValueError('unsafe archive path')
            if relative.name == 'Thumbs.db':
                continue  # Original ingestion intentionally did not extract this cache.
            target = (original_root / Path(*relative.parts[1:])).resolve()
            if not target.is_relative_to(original_root):
                raise ValueError('original path escapes source directory')
            if target.read_bytes() != archive.read(name):
                raise ValueError('extracted original differs from archive: ' + name)
            paths.append(target)
    return [binding(path) for path in paths]


def exif_diagnostics(images):
    from PIL import Image

    fields = {271: 'make', 272: 'model', 33434: 'exposure_time', 33437: 'f_number',
              34855: 'iso', 37380: 'exposure_bias'}
    records = []
    for row in images:
        path = Path(row['path']).resolve()
        if not path.is_relative_to((SOURCE / 'originals').resolve()) or digest(path) != row['sha256']:
            raise ValueError('image path/hash changed')
        with Image.open(path) as picture:
            exif = picture.getexif()
            values = dict(exif)
            values.update(exif.get_ifd(34665))
            found = {name: str(values[tag]) for tag, name in fields.items() if tag in values}
            records.append(dict(subject_id=row['subject_id'], image_id=row['image_id'],
                                has_exif=bool(exif), has_icc=bool(picture.info.get('icc_profile')),
                                capture_fields=found, source_camera=row['camera'],
                                source_exposure=row['exposure']))
    return dict(records=records, images_with_exif=sum(row['has_exif'] for row in records),
                images_with_icc=sum(row['has_icc'] for row in records),
                missing_camera=sum(row['source_camera'] is None for row in records),
                missing_exposure=sum(row['source_exposure'] is None for row in records),
                inferred_or_repaired_capture_labels=0)


def prepare_result():
    if digest(SOURCE / 'profile.json') != PROFILE_SHA or digest(PALETTE) != PALETTE_SHA:
        raise ValueError('source profile changed; version this analysis instead of replacing sources')
    profile = read_json(SOURCE / 'profile.json')
    inputs = audit_originals(SOURCE, profile) + [binding(PALETTE)]
    documents = []
    for name, expected in DOCUMENTS.items():
        path = OUT / 'sources' / name
        receipt_path = path.with_name(name + '.source.json')
        receipt = read_json(receipt_path)
        if digest(path) != expected or receipt['sha256'] != expected or receipt['bytes'] != path.stat().st_size:
            raise ValueError('source document or receipt changed: ' + name)
        inputs.extend([binding(path), binding(receipt_path)])
        documents.append(dict(file=name, url=receipt['url'], sha256=expected))
    whites = {name: [100*v for v in value]
              for name, value in read_json(PALETTE)['white_xyz_y1'].items()}
    analysis = qualify_profile(profile, whites)
    analysis['capture_metadata_check'] = exif_diagnostics(profile['images'])
    inputs.extend([binding(Path(__file__)), binding(ROOT / 'tests/test_skin_dast_qualify.py')])
    for item in inputs:
        if digest(item['path']) != item['sha256']:
            raise ValueError('input changed during the read-only audit')
    return dict(schema='luma.dast.qualification.v1', documented_convention=CONVENTION,
                analysis=analysis, source_documents=documents, inputs=inputs,
                source_preservation='all extracted non-cache originals replayed byte-for-byte against original ZIP; hashes rechecked',
                prior_model_sources_modified=False, models_run=0, new_photographs=0,
                new_measurements=0, new_training_targets=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'verify'])
    args = parser.parse_args()
    target = OUT / 'qualification.json'
    if args.command == 'run' and target.exists():
        raise FileExistsError('qualification already exists; use verify')
    result = prepare_result()
    if args.command == 'run':
        result['created_utc'] = datetime.now(timezone.utc).isoformat()
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
    else:
        saved = read_json(target)
        saved.pop('created_utc')
        if saved != result:
            raise ValueError('qualification does not reproduce from preserved inputs')
    print(json.dumps(dict(command=args.command, qualification_sha256=digest(target),
                          counts=result['analysis']['counts'], inputs_checked=len(result['inputs']),
                          numeric={key: {k: v for k, v in value.items() if k != 'per_measurement_residual'}
                                   for key, value in result['analysis']['white_diagnostic']['candidates'].items()},
                          capture={key: value for key, value in result['analysis']['capture_metadata_check'].items()
                                   if key != 'records'}), ensure_ascii=False))


if __name__ == '__main__':
    main()
