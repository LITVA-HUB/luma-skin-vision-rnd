"""Fetch official public ENCoDE documentation; never authenticate or accept DUA."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

SOURCES = {
    'physionet_page.html': 'https://physionet.org/content/encode-skin-color/1.0.0/',
    'files_gate.html': 'https://physionet.org/files/encode-skin-color/1.0.0/',
    'license.html': 'https://physionet.org/content/encode-skin-color/view-license/1.0.0/',
    'dua.html': 'https://physionet.org/content/encode-skin-color/view-dua/1.0.0/',
    'README.md': 'https://raw.githubusercontent.com/aiwonglab/ENCoDE_tutorial/master/README.md',
    'LICENSE': 'https://raw.githubusercontent.com/aiwonglab/ENCoDE_tutorial/master/LICENSE',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--use-existing', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        destination = args.output / name
        if args.use_existing and destination.exists():
            content = destination.read_bytes()
            status = None  # Existing HTTP evidence remains in original access_probe.json.
            final = url
            obtained = 'existing_current_session_download'
        else:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    content, status, final = response.read(), response.status, response.url
            except urllib.error.HTTPError as error:
                content, status, final = error.read(), error.code, error.url
            destination.write_bytes(content)
            obtained = 'new_request'
        records.append({'file': name, 'url': url, 'final_url': final, 'http_status': status,
                        'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                        'obtained': obtained})
    receipt = {'dataset': 'ENCoDE', 'version': '1.0.0',
               'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'authenticated': False, 'accepted_terms': False,
               'participant_files_downloaded': 0, 'images_downloaded': 0,
               'instrument_pairs_verified': 0, 'files': records}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'public_document_files': len(records), 'participant_files': 0}))


if __name__ == '__main__':
    main()
