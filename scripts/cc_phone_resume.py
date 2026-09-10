"""Polite transport wrapper; frozen phone selection/CRC/size logic is unchanged."""
import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError


def polite_opener(original, sleep=time.sleep):
    def fetch(*args, **kwargs):
        for attempt in range(5):
            sleep(1.25)
            try:
                return original(*args, **kwargs)
            except HTTPError as exc:
                if exc.code not in (429, 503) or attempt == 4:
                    raise
                value = exc.headers.get('Retry-After', '60')
                try:
                    delay = float(value)
                except ValueError:
                    delay = (parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds()
                delay = max(60*(attempt+1), delay)
                print(json.dumps({'http_status': exc.code, 'retry_after_seconds': delay}), flush=True)
                exc.close()
                sleep(delay)
        raise RuntimeError('Unreachable retry state')
    return fetch


if __name__ == '__main__':
    import cc_phone_data

    from luma_skin_vision.data import sha256
    from luma_skin_vision.experiment import write_json

    parser = argparse.ArgumentParser()
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--out', default=str(cc_phone_data.ROOT/'data/public/beyond_rgb_phone'))
    parser.add_argument('--role', default='all', choices=('all', 'loader', 'reserved_test'))
    args = parser.parse_args()
    write_json(cc_phone_data.PROVENANCE/'phone_resume_transport.json', {
        'wrapper_sha256': sha256(__file__), 'frozen_download_sha256': sha256(cc_phone_data.__file__),
        'lock_sha256': args.lock_sha256, 'interval_seconds': 1.25,
        'retry_statuses': [429, 503], 'max_attempts': 5,
        'note': 'Same source URLs/ranges/selection; existing files reverified and skipped; respect Retry-After.'})
    urllib.request.urlopen = polite_opener(urllib.request.urlopen)
    cc_phone_data.download(args)
