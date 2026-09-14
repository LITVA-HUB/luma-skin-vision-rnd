"""Download the author's public LaPa archive, keeping its original bytes and receipt."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from skin_data_growth import DATA_ROOT, fetch_original, sha_bytes, write_json


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields, self.active = None, {}, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and a.get('id') == 'download-form':
            self.action, self.active = a.get('action'), True
        if self.active and tag == 'input' and a.get('type') == 'hidden':
            self.fields[a['name']] = a['value']

    def handle_endtag(self, tag):
        if tag == 'form':
            self.active = False


def main():
    root = DATA_ROOT / 'lapa'
    archive = root/'LaPa.tar.gz'
    receipt_path = root/'archive_source.json'
    if archive.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        with archive.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == receipt['sha256']
        print('LAPA ORIGINAL VERIFIED', receipt['bytes'])
        return
    file_id = '1XOBoRGSraP50_pS1YPB8_i8Wmw_5L-NG'
    landing_url = 'https://drive.google.com/uc?export=download&id='+file_id
    fetch_original(landing_url, root/'download_landing.html')
    form = DownloadForm()
    form.feed((root/'download_landing.html').read_text(encoding='utf-8'))
    if form.action != 'https://drive.usercontent.google.com/download' or form.fields.get('id') != file_id:
        raise ValueError('unexpected public download form')
    url = form.action + '?' + urllib.parse.urlencode(form.fields)
    partial = root/'LaPa.tar.gz.partial'
    if partial.exists():
        raise RuntimeError('partial acquisition exists; inspect before resuming')
    h, count, started, tick = hashlib.sha256(), 0, time.perf_counter(), time.perf_counter()
    request = urllib.request.Request(url, headers={'User-Agent':'Luma-Research'})
    with urllib.request.urlopen(request, timeout=45) as response:
        expected = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
        if expected is not None and expected > 3_000_000_000:
            raise ValueError('archive exceeds acquisition bound')
        first = response.read(1024*1024)
        if first[:2] != b'\x1f\x8b':
            raise ValueError('download is not gzip data; no HTML/error saved as archive')
        with partial.open('xb') as f:
            chunk = first
            while chunk:
                count += len(chunk)
                if count > 3_000_000_000:
                    raise ValueError('archive exceeds acquisition bound')
                f.write(chunk)
                h.update(chunk)
                if time.perf_counter()-tick > 5:
                    print('LAPA DOWNLOAD',count,'of',expected,'MB/s',round(count/(time.perf_counter()-started)/1e6,2),flush=True)
                    tick = time.perf_counter()
                chunk = response.read(1024*1024)
        if expected is not None and count != expected:
            raise ValueError('truncated archive')
        receipt = dict(source='https://github.com/jd-opensource/lapa-dataset',
                       landing_url=landing_url, bytes=count, sha256=h.hexdigest(),
                       seconds=time.perf_counter()-started,
                       content_type=response.headers.get('Content-Type'), etag=response.headers.get('ETag'),
                       acquisition_code_sha256=sha_bytes(Path(__file__).read_bytes()),
                       original_readme_sha256=sha_bytes((root/'README.md').read_bytes()),
                       license='author non-commercial research/personal experimentation terms; original LICENSE retained',
                       label_kind='semantic face parsing labels and landmarks; no measured skin-color truth',
                       images_validated=False)
    write_json(receipt_path, receipt)
    # Both checked absolute paths are siblings inside this task's D: data directory.
    assert partial.resolve().parent == archive.resolve().parent == root.resolve()
    partial.rename(archive)
    print('LAPA DOWNLOAD COMPLETE',json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
