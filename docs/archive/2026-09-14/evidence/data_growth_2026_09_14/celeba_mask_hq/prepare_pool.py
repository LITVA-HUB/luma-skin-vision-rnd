"""Audit original face masks and create an unsplit auxiliary pool; no training."""
import collections
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / 'CelebAMask-HQ.zip'
OUT = ROOT / 'prepared_192_v1'
LABELS = {'skin', 'nose', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear',
          'r_ear', 'ear_r', 'u_lip', 'l_lip', 'mouth', 'hair', 'hat', 'neck', 'neck_l', 'cloth'}
POSITIVE = {'skin', 'nose'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def decode_mask(raw):
    with Image.open(io.BytesIO(raw)) as im:
        if im.format != 'PNG' or im.size != (512, 512):
            raise ValueError('Unexpected mask image format/size')
        a = np.array(im)
    if a.dtype != np.uint8 or a.shape != (512, 512, 3):
        raise ValueError('Expected author RGB binary mask')
    if not np.array_equal(a[:, :, 0], a[:, :, 1]) or not np.array_equal(a[:, :, 0], a[:, :, 2]):
        raise ValueError('Mask channels disagree')
    gray = a[:, :, 0]
    if np.any((gray != 0) & (gray != 255)):
        raise ValueError('Mask is not binary')
    return gray != 0


def main():
    started = time.perf_counter()
    receipt = json.loads((ROOT / 'archive_source.json').read_text())
    if sha(ARCHIVE) != receipt['sha256']:
        raise ValueError('Archive checksum differs')
    OUT.mkdir(exist_ok=False)
    contract = dict(archive_sha256=receipt['sha256'], script_sha256=sha(__file__),
                    role='auxiliary_pool_unassigned', instrument_lab_targets=False,
                    skin_rule='union(skin,nose) minus union(all 16 other author foreground classes)',
                    mask_channels='RGB channels must be equal and contain only 0/255',
                    image_resize='Pillow BILINEAR to 192x192, uint8 RGB HWC',
                    mask_resize='512x512 binary union/exclusion, Pillow NEAREST to 192x192',
                    evaluation='No split, fitting, model selection, or quality evaluation in this acquisition',
                    source_people='Not inferred. Original image mapping is not a person identifier.',
                    frozen_HR_P3_inputs_changed=False)
    save_json(OUT / 'protocol.json', contract)
    rows, mask_totals, signatures = [], collections.Counter(), collections.defaultdict(list)
    image_shapes = collections.Counter()
    with zipfile.ZipFile(ARCHIVE) as z:
        infos = z.infolist()
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError('Duplicate archive member names')
        images, masks = {}, collections.defaultdict(dict)
        for item in infos:
            p = PurePosixPath(item.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in item.filename or ':' in item.filename:
                raise ValueError('Unsafe member path')
            if item.is_dir():
                continue
            if item.file_size > 20_000_000:
                raise ValueError('Oversized member')
            if p.name == '.DS_Store':
                z.read(item)  # Author OS sidecar, verified but never a training example.
                continue
            if '/CelebA-HQ-img/' in item.filename:
                if not re.fullmatch(r'\d+\.jpg', p.name):
                    raise ValueError('Unknown image member')
                index = int(p.stem)
                if index in images:
                    raise ValueError('Duplicate image index')
                images[index] = item.filename
            elif '/CelebAMask-HQ-mask-anno/' in item.filename:
                match = re.fullmatch(r'(\d{5})_(.+)\.png', p.name)
                if match is None or match[2] not in LABELS:
                    raise ValueError('Unknown mask member')
                index, label = int(match[1]), match[2]
                if label in masks[index] or int(p.parent.name) != index // 2000:
                    raise ValueError('Duplicate mask or wrong source directory')
                masks[index][label] = item.filename
            else:
                # Integrity-check source README and mapping/annotation tables too; do not use attribute labels.
                z.read(item)
        if set(images) != set(range(30000)) or set(masks) != set(images):
            raise ValueError('Missing or orphan image/mask index')
        if any('skin' not in m for m in masks.values()):
            raise ValueError('Missing primary skin annotation')
        mapping_name = 'CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt'
        mapping_raw = z.read(mapping_name)
        mapping = {}
        for line in mapping_raw.decode().splitlines()[1:]:
            i, original_index, original_file = line.split()
            if int(i) in mapping or int(original_index) + 1 != int(Path(original_file).stem):
                raise ValueError('Invalid original image mapping')
            mapping[int(i)] = dict(original_index=int(original_index), original_file=original_file)
        if set(mapping) != set(images) or len({m['original_file'] for m in mapping.values()}) != 30000:
            raise ValueError('Incomplete or duplicate original image mapping')
        (OUT / 'original_image_mapping.txt').write_bytes(mapping_raw)
        rgb = np.lib.format.open_memmap(OUT / 'images.npy', mode='w+', dtype=np.uint8,
                                       shape=(30000, 192, 192, 3))
        target = np.lib.format.open_memmap(OUT / 'face_skin.npy', mode='w+', dtype=np.uint8,
                                          shape=(30000, 192, 192))
        for index in range(30000):
            raw = z.read(images[index])  # zipfile checks CRC on the original compressed member.
            digest = hashlib.sha256(raw).hexdigest()
            signatures[digest].append(index)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format != 'JPEG' or im.mode != 'RGB' or im.size != (1024, 1024):
                    raise ValueError('Unexpected original photo format, mode or size')
                im.load()  # fully decode, including the original JPEG payload
                image_shapes[str(im.size)] += 1
                rgb[index] = np.asarray(im.resize((192, 192), Image.Resampling.BILINEAR))
            positive = np.zeros((512, 512), dtype=bool)
            excluded = np.zeros_like(positive)
            component_hashes = {}
            for label, name in sorted(masks[index].items()):
                component = z.read(name)
                component_hashes[label] = hashlib.sha256(component).hexdigest()
                binary = decode_mask(component)
                if label in POSITIVE:
                    positive |= binary
                else:
                    excluded |= binary
                mask_totals[label] += 1
            clean = positive & ~excluded
            scaled = np.asarray(Image.fromarray(clean.astype(np.uint8)).resize(
                (192, 192), Image.Resampling.NEAREST))
            target[index] = scaled
            rows.append(dict(index=index, source_member=images[index], source_sha256=digest,
                             **mapping[index], mask_sha256=component_hashes,
                             annotated_skin_pixels_512=int(positive.sum()),
                             excluded_overlap_pixels_512=int((positive & excluded).sum()),
                             face_skin_pixels_512=int(clean.sum()), face_skin_pixels_192=int(scaled.sum()),
                             present_labels=sorted(masks[index]), instrument_lab=None,
                             person_id=None, role='auxiliary_pool_unassigned'))
            if (index + 1) % 1000 == 0:
                rgb.flush(); target.flush()
                print(json.dumps({'prepared_images': index + 1, 'total_images': 30000,
                                  'seconds': time.perf_counter() - started}), flush=True)
        rgb.flush(); target.flush()
        del rgb, target
    duplicates = [ids for ids in signatures.values() if len(ids) > 1]
    lapa_manifest_path = ROOT.parent / 'lapa/file_manifest.json'
    lapa = json.loads(lapa_manifest_path.read_text())
    lapa_rows = lapa['records']
    lapa_hashes = {r['sha256'] for r in lapa_rows if r.get('kind') == 'images'}
    exact_overlap = sorted(i for digest in lapa_hashes & set(signatures) for i in signatures[digest])
    save_json(OUT / 'rows.json', rows)
    profile = dict(source_images=30000, source_mask_pngs=sum(mask_totals.values()),
                   class_mask_counts=dict(sorted(mask_totals.items())), source_photo_shapes=dict(image_shapes),
                   all_source_images_and_masks_decoded=True, all_archive_member_crcs_checked=True,
                   exact_duplicate_image_groups=duplicates,
                   unique_image_byte_hashes=len(signatures),
                   exact_byte_overlap_with_lapa_ids=exact_overlap,
                   lapa_manifest_sha256=sha(lapa_manifest_path),
                   overlap_limit='Exact JPEG bytes only; recropped/reencoded images or same-person overlap not ruled out.',
                   empty_face_masks_192=[r['index'] for r in rows if r['face_skin_pixels_192'] == 0],
                   images_with_positive_exclusion_overlap=sum(r['excluded_overlap_pixels_512'] > 0 for r in rows),
                   total_removed_pixels_512=sum(r['excluded_overlap_pixels_512'] for r in rows),
                   no_nose_mask_ids=[r['index'] for r in rows if 'nose' not in r['present_labels']],
                   mask_fraction_quantiles=np.quantile([r['face_skin_pixels_192']/192**2 for r in rows],
                                                       [0, .01, .5, .99, 1]).tolist(),
                   ground_truth_type='manual binary facial skin mask; no measured skin-color label',
                   source_person_count=None, role='auxiliary_pool_unassigned',
                   preparation_seconds=time.perf_counter() - started,
                   files={name:dict(bytes=(OUT/name).stat().st_size, sha256=sha(OUT/name))
                          for name in ['images.npy', 'face_skin.npy', 'rows.json', 'protocol.json', 'original_image_mapping.txt']})
    save_json(OUT / 'profile.json', profile)
    print('POOL_PREPARATION_COMPLETE', json.dumps(profile), flush=True)


if __name__ == '__main__':
    main()
