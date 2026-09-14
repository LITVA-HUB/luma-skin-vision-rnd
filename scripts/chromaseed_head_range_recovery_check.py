"""Observe the replayed checkpoint and preserved banks without touching training."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

RUN = Path('D:/Luma-RnD/chromaseed_head_range_v1')
OUT = RUN / 'recovery_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    previous = read(OUT / 'failed_progress.json')
    recovery = read(OUT / 'protocol.json')
    launch = read(OUT / 'launch.json')
    contract = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
                    recovery_protocol_sha256=sha(OUT / 'protocol.json'),
                    previous_progress_sha256=sha(OUT / 'failed_progress.json'),
                    expected_step=1536, timeout_seconds=600, poll_seconds=5,
                    check='Exact same-step six-slot loss and unchanged previous 18 banks; no quality selection')
    save(OUT / 'check_protocol.json', contract)
    if previous['step'] != 1536 or previous['variant'] != 'soft5m':
        raise ValueError('Unexpected failed-run boundary')
    bank = RUN / 'inner/mixed/soft5m__wide/fold0'
    same_step = None
    maximum_step = 0
    deadline = time.monotonic()+contract['timeout_seconds']
    while time.monotonic() < deadline:
        with urlopen('http://127.0.0.1:8766/api/status', timeout=10) as stream:
            snapshot = json.load(stream)['head_range']
        current = snapshot['current']
        if snapshot['state'] in ('failed', 'interrupted'):
            raise RuntimeError('Resumed primary is no longer live: ' + str(snapshot.get('error')))
        matches = current.get('pid') == launch['pid'] and all(
            current.get(k) == previous[k] for k in ('role', 'variant', 'head_mode', 'fold', 'stage'))
        if matches:
            maximum_step = max(maximum_step, current.get('step', 0))
            if current.get('step') == previous['step'] and same_step is None:
                if current['minibatch_loss'] != previous['minibatch_loss']:
                    raise ValueError('Replayed same-step loss is not exactly identical')
                same_step = dict(step=previous['step'], original_losses=previous['minibatch_loss'],
                                 replayed_losses=current['minibatch_loss'], exact=True)
                print('HR REPLAY', previous['step'], 'six-slot loss exact', flush=True)
        if (bank / 'receipt.json').exists():
            receipt = read(bank / 'receipt.json')
            if receipt['source_lock_sha256'] != recovery['original_source_lock_sha256'] or receipt['steps'] != 2048:
                raise ValueError('Resumed bank protocol/steps changed')
            for path, expected in recovery['completed_bank_files'].items():
                if sha(path) != expected:
                    raise ValueError('A previously complete bank changed: ' + path)
            for name, expected in receipt['files'].items():
                if sha(bank / name) != expected:
                    raise ValueError('Newly completed bank checksum mismatch')
            if sha(__file__) != contract['source_sha256']:
                raise ValueError('Observer source changed during checking')
            result = dict(check_protocol_sha256=sha(OUT / 'check_protocol.json'),
                          status='passed', previous_banks_unchanged=18,
                          preserved_files_checked=len(recovery['completed_bank_files']),
                          replayed_same_step=same_step, same_step_observation_missed=same_step is None,
                          maximum_progress_step_observed=maximum_step,
                          resumed_bank_receipt_sha256=sha(bank / 'receipt.json'),
                          resumed_bank_files_checked=len(receipt['files']),
                          resumed_bank_steps=receipt['steps'], original_sources_unchanged=True,
                          quality_improvement_claim=False, full_primary_audit=False)
            save(OUT / 'check.json', result)
            print('HR RECOVERY CHECK COMPLETE', json.dumps(result), flush=True)
            return
        time.sleep(contract['poll_seconds'])
    raise TimeoutError('No completed replayed bank within observer window; primary is not terminated')


if __name__ == '__main__':
    main()
