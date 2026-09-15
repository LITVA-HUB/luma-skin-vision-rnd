"""Restore exact historical MSKCC release; never overwrite a mismatching cache."""
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
receipt = json.loads((root/'docs/data/provenance/mskcc_skin_v1/acquisition.json').read_text())
raw = root/'data/public/mskcc_skin_v1'
raw.mkdir(parents=True, exist_ok=True)
for rec in receipt['files']:
    out = raw/rec['file']
    cached = out.exists()
    if cached:
        body = out.read_bytes()
    else:
        if not rec['url'].startswith('https://isic-archive.s3.amazonaws.com/dois/10.34970-962049/'):
            raise ValueError('Unexpected source URL')
        with urllib.request.urlopen(rec['url'], timeout=60) as response:
            body = response.read(rec['bytes']+1)
    if len(body) != rec['bytes'] or hashlib.sha256(body).hexdigest() != rec['sha256']:
        raise ValueError('Historical source mismatch; nothing overwritten: '+rec['file'])
    if not cached:
        with tempfile.NamedTemporaryFile(dir=raw, suffix='.part', delete=False) as stream:
            stream.write(body); temporary = Path(stream.name)
        temporary.replace(out)
    print(json.dumps({'file': rec['file'], 'bytes': len(body), 'sha256': rec['sha256'],
                      'verified': True, 'reused_cache': cached}), flush=True)
