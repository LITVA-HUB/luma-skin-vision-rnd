"""Download the author-linked public archive; preserve original bytes and receipt."""
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = Path('C:/Users/dimal/Documents/просто/.worktrees/luma-local-search/scripts')
sys.path.insert(0, str(SCRIPTS))
from skin_lapa_acquire import DownloadForm

ARCHIVE = ROOT / 'CelebAMask-HQ.zip'
PARTIAL = ROOT / 'CelebAMask-HQ.zip.partial'
RECEIPT = ROOT / 'archive_source.json'
if any(p.exists() for p in (ARCHIVE, PARTIAL, RECEIPT)):
    raise RuntimeError('Existing acquisition must be inspected; no overwrite or restart')
form = DownloadForm()
form.feed((ROOT / 'download_landing.html').read_text())
assert form.action == 'https://drive.usercontent.google.com/download'
assert form.fields['id'] == '1badu11NqxGf6qM3PTTooQDJvQbejgbTv'
url = form.action + '?' + urllib.parse.urlencode(form.fields)
request = urllib.request.Request(url, headers={'User-Agent': 'LumaResearch/1.0'})
digest, count, started, tick = hashlib.sha256(), 0, time.perf_counter(), time.perf_counter()
with urllib.request.urlopen(request, timeout=45) as response:
    expected = int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
    if expected is not None and expected > 6_000_000_000:
        raise ValueError('Archive exceeds local acquisition bound')
    first = response.read(4 * 1024 * 1024)
    if first[:4] != b'PK\x03\x04':
        raise ValueError('Not a ZIP stream; no HTML/error saved as data')
    with PARTIAL.open('xb') as output:
        chunk = first
        while chunk:
            count += len(chunk)
            if count > 6_000_000_000:
                raise ValueError('Archive exceeds local acquisition bound')
            output.write(chunk)
            digest.update(chunk)
            if time.perf_counter() - tick > 15:
                print(json.dumps({'download_bytes': count, 'expected_bytes': expected,
                                  'MB_per_second': round(count / (time.perf_counter() - started) / 1e6, 2)}), flush=True)
                tick = time.perf_counter()
            chunk = response.read(4 * 1024 * 1024)
    if expected is not None and count != expected:
        raise ValueError('Incomplete stream')
    receipt = dict(source='https://github.com/switchablenorms/CelebAMask-HQ',
                   author_archive_url='https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv',
                   acquired_utc=datetime.now(timezone.utc).isoformat(), bytes=count,
                   sha256=digest.hexdigest(), download_seconds=time.perf_counter() - started,
                   content_type=response.headers.get('Content-Type'), etag=response.headers.get('ETag'),
                   acquisition_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   original_readme_sha256=hashlib.sha256((ROOT/'README.md').read_bytes()).hexdigest(),
                   label_kind='Manual semantic face masks. No instrument-measured skin-color target.',
                   original_terms='Author non-commercial research terms retained in README.md; not an unrestricted grant.',
                   images_validated=False, entered_registered_HR_or_P3=False)
assert PARTIAL.resolve().parent == ARCHIVE.resolve().parent == ROOT
PARTIAL.rename(ARCHIVE)
with RECEIPT.open('x', encoding='utf-8') as output:
    json.dump(receipt, output, ensure_ascii=False, indent=2)
    output.write('\n')
print('ACQUISITION_COMPLETE', json.dumps(receipt), flush=True)
