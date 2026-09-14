"""Resume frozen HR after a Windows status-file sharing failure; I/O-only adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path('D:/Luma-RnD/chromaseed_head_range_v1')
OUT = RUN / 'recovery_v1'
SOURCE_LOCK = 'dfb96f902658009f57d37478ee9e2964c951a2bbce8aaba9acda77ff142652b2'
DELAYS = (.01, .02, .04, .08, .16, .32) + (.5,)*10


def retry_writer(writer, progress_path, *, delays=DELAYS, sleep=time.sleep, report=None):
    """Retry denied replacement; only the exact progress file can be skipped."""
    progress_path = Path(progress_path).resolve()
    report = report or (lambda event: print('HR IO', json.dumps(event), flush=True))

    def write(path, value):
        path = Path(path)
        for attempt in range(len(delays)+1):
            try:
                result = writer(path, value)
            except PermissionError as exc:
                if attempt < len(delays):
                    sleep(delays[attempt])
                    continue
                noncritical = path.resolve() == progress_path
                report(dict(path=str(path), retries=attempt, step=value.get('step'),
                            outcome='telemetry_skipped' if noncritical else 'critical_failed',
                            error=str(exc)))
                if noncritical:
                    return None
                raise
            else:
                if attempt:
                    report(dict(path=str(path), retries=attempt, step=value.get('step'), outcome='retried'))
                return result
        raise AssertionError('Unreachable retry state')

    return write


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_once(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def original_sources():
    if sha(RUN / 'source_lock.json') != SOURCE_LOCK:
        raise ValueError('Frozen HR source lock changed')
    source = read(RUN / 'source_lock.json')
    for name, expected in source['sources'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Immutable numerical/protocol source changed: ' + name)
    return source


def freeze():
    original_sources()
    job = read(RUN / 'job.json')
    if job.get('status') != 'failed' or job.get('completed_banks') != 18 or 'WinError 5' not in job.get('error', ''):
        raise ValueError('Recovery is specific to the recorded 18-bank Windows failure')
    if (RUN / 'results.json').exists() or (ROOT / 'docs/benchmarks/chromaseed_head_range_v1/verification.json').exists():
        raise ValueError('Do not resume completed or sealed work')
    banks = sorted(RUN.glob('inner/*/*/*/receipt.json'))
    if len(banks) != 18 or any(RUN.glob('final/*/*/*/receipt.json')):
        raise ValueError('Unexpected completion boundary')
    bindings = {}
    for receipt in banks:
        value = read(receipt)
        if value['source_lock_sha256'] != SOURCE_LOCK:
            raise ValueError('Existing bank has a different experimental protocol')
        bindings[str(receipt)] = sha(receipt)
        for name, expected in value['files'].items():
            path = receipt.parent / name
            if sha(path) != expected:
                raise ValueError('Completed bank artifact changed')
            bindings[str(path)] = expected
    OUT.mkdir(parents=True, exist_ok=True)
    preserved = {}
    for name in ('job.json', 'progress.json', 'progress.tmp', 'production_v1.log'):
        path = RUN / name
        if path.exists():
            target = OUT / ('failed_' + name)
            if target.exists():
                raise ValueError('Preserve previous recovery evidence')
            shutil.copy2(path, target)
            if sha(target) != sha(path):
                raise ValueError('Failure evidence copy mismatch')
            preserved[str(target)] = sha(target)
    sources = [Path(__file__), ROOT / 'tests/test_chromaseed_head_range_recovery.py',
               ROOT / 'docs/research/chromaseed_head_range_recovery_v1.md']
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), original_source_lock_sha256=SOURCE_LOCK,
                  source_bindings={str(p): sha(p) for p in sources}, preserved_failure=preserved,
                  completed_bank_files=bindings, completed_banks=18,
                  numerical_change=False, retry_delays_seconds=list(DELAYS),
                  nonfatal_only=str((RUN / 'progress.json').resolve()),
                  unfinished_bank='mixed/soft5m/wide/fold0; restart from its original fixed initialization',
                  initial_attempt_terminal_exit_code=1)
    save_once(OUT / 'protocol.json', record)
    print('HR RECOVERY FROZEN', sha(OUT / 'protocol.json'), flush=True)


def resume():
    original_sources()
    recovery = read(OUT / 'protocol.json')
    for group in ('source_bindings', 'preserved_failure', 'completed_bank_files'):
        for path, expected in recovery[group].items():
            if sha(path) != expected:
                raise ValueError('Recovery binding changed: ' + path)
    # Same explicit host-thread settings as the original launch, before Torch import.
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    import chromaseed_head_range_run as runner

    if (RUN / 'results.json').exists() or read(RUN / 'job.json').get('status') != 'failed':
        raise ValueError('Do not launch a duplicate or completed HR primary')
    event_path = OUT / 'io_events.jsonl'

    def report(event):
        event = dict(event, utc=datetime.now(timezone.utc).isoformat())
        print('HR IO', json.dumps(event, ensure_ascii=False), flush=True)
        try:
            with event_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False)+'\n')
        except OSError as exc:
            print('HR IO EVENT LOG UNAVAILABLE', str(exc), flush=True)

    unchanged = {name: getattr(runner, name) for name in ('fit', 'predict_torch', 'train_bank', 'select', 'evaluate', 'save')}
    runner.write_json = retry_writer(runner.write_json, RUN / 'progress.json', report=report)
    assert all(getattr(runner, name) is value for name, value in unchanged.items())
    save_once(OUT / 'launch.json', dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(),
              recovery_protocol_sha256=sha(OUT / 'protocol.json'),
              original_source_lock_sha256=SOURCE_LOCK, adapter='runner.write_json only',
              numerical_callables_unchanged=list(unchanged)))
    # The original parser accepts no recovery positional argument.
    import sys
    sys.argv = [str(ROOT / 'scripts/chromaseed_head_range_run.py')]
    runner.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'resume'])
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        resume()
