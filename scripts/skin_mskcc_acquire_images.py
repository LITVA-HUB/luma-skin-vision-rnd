"""Acquire only instrument-paired original images using licensed ISIC API records."""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from skin_mskcc_data import RAW, PROV, manifest, sha, MANIFEST


def fetch(url, cap):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read(cap + 1)
            if len(data) > cap:
                raise ValueError("Response exceeds acquisition limit")
            return data
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def one(row):
    image_id = row['image']
    api_path = RAW / 'api' / (image_id + '.json')
    if not api_path.exists():
        data = fetch('https://api.isic-archive.com/api/v2/images/' + image_id + '/', 100000)
        info = json.loads(data)
        if info['copyright_license'] != 'CC-BY' or info['isic_id'] != image_id:
            raise ValueError("Original image license or identity mismatch")
        api_path.write_bytes(data)
    info = json.loads(api_path.read_bytes())
    if info['copyright_license'] != 'CC-BY' or info['isic_id'] != image_id:
        raise ValueError("Cached image license or identity mismatch")
    item = info['files']['full']
    if not item['url'].startswith('https://isic-archive.s3.amazonaws.com/images/'):
        raise ValueError("Unrecognized original host")
    if not 1 <= item['size'] <= 15_000_000:
        raise ValueError("Unexpected image size")
    path = RAW / 'images' / (image_id + '.jpg')
    header_path = RAW / 'api' / (image_id + '_s3.json')
    if not path.exists():
        # ISIC API sizes omit bytes present in current S3 objects (observed
        # 896194 versus 899367). Validate the actual original HTTP object and
        # its S3 single-part MD5, retaining the discrepancy explicitly.
        with urllib.request.urlopen(item['url'], timeout=45) as response:
            headers = dict(response.headers)
            data = response.read(15_000_001)
        length = int(headers['Content-Length'])
        etag = headers['ETag'].strip('"')
        if len(data) != length or length > 15_000_000 or not data.startswith(b'\xff\xd8'):
            raise ValueError('Invalid original JPEG or HTTP length')
        if len(etag) != 32 or hashlib.md5(data).hexdigest() != etag:
            raise ValueError('Original single-part S3 ETag MD5 mismatch')
        header_path.write_text(json.dumps(headers, indent=2)+'\n', encoding='utf8')
        tmp = path.with_suffix('.part')
        tmp.write_bytes(data)
        tmp.replace(path)
    headers = json.loads(header_path.read_bytes())
    if path.stat().st_size != int(headers['Content-Length']) or hashlib.md5(path.read_bytes()).hexdigest() != headers['ETag'].strip('"'):
        raise ValueError("Cached original image differs from S3 receipt")
    return {'image': image_id, 'bytes': path.stat().st_size, 'api_declared_bytes': item['size'], 'sha256': sha(path),
            'api_sha256': sha(api_path), 'url': item['url'], 'license': info['copyright_license']}


def main():
    rows = manifest()['rows']
    if shutil.disk_usage(RAW).free < 12_000_000_000:
        raise ValueError("Need 12 GB free for bounded acquisition")
    for d in ['api', 'images']:
        (RAW / d).mkdir(exist_ok=True)
    receipts = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(one, r) for r in rows]
    try:
        for future in concurrent.futures.as_completed(futures):
            receipts.append(future.result())
            if sum(r['bytes'] for r in receipts) > 6_000_000_000:
                raise ValueError('Subset acquisition exceeded 6 GB budget')
            if len(receipts) % 100 == 0:
                print(json.dumps({'acquired': len(receipts), 'total': len(rows),
                                  'bytes': sum(r['bytes'] for r in receipts)}), flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    receipts.sort(key=lambda r: r['image'])
    local = RAW / 'image_receipts.json'
    local.write_text(json.dumps(receipts, indent=2) + '\n', encoding='utf8')
    summary = {'utc': datetime.now(timezone.utc).isoformat(), 'original_images': len(receipts),
               'total_bytes': sum(r['bytes'] for r in receipts), 'license': 'CC-BY each original API record',
               'manifest_sha256': sha(MANIFEST), 'local_image_receipts_sha256': sha(local),
               'api_s3_size_discrepancies': sum(r['api_declared_bytes'] != r['bytes'] for r in receipts),
               'integrity': 'Every JPEG checked against original S3 Content-Length and single-part ETag MD5',
               'pixels_decoded': False, 'full_5_85_GB_bundle_downloaded': False}
    (PROV / 'image_acquisition.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
