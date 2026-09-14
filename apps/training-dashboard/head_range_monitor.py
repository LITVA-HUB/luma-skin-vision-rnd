"""Read-only HR status and work estimates; no ML dependencies or writers."""
import json
import statistics
import time
from pathlib import Path

from monitor import LABELS, ROLES, VARIANTS, process_alive


def snapshot(root=Path('D:/Luma-RnD/chromaseed_head_range_v1'), now=None, probe=process_alive):
    root = Path(root)
    now = time.time() if now is None else now

    def read(path, default=None):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return default

    if not (root/'source_lock.json').exists():
        return None
    job, progress = read(root/'job.json', {}), read(root/'progress.json', {})
    selection = read(root/'selections.json')
    state = job.get('status', 'ready')
    if state == 'running':
        live = probe(job.get('pid'))
        state = progress.get('status','training') if live is True else 'interrupted' if live is False else 'unknown'
        path = root/'progress.json'
        if live is True and path.exists() and now-path.stat().st_mtime > 180:
            state = 'stale'
    entries, rates = [], {}
    for role in ROLES:
        for variant in VARIANTS:
            for mode in ['wide','linear']:
                pair = variant+'__'+mode
                for stage,fold in [('inner',f) for f in range(3)]+[('final',None)]:
                    path = root/stage/role/pair/('bank' if fold is None else f'fold{fold}')/'receipt.json'
                    receipt = read(path)
                    target = 2048 if fold is not None or not selection else selection['roles'][role]['policies']['per_pair'][pair]['step']
                    key = (stage,role,variant,mode,fold)
                    entries.append((key,target,receipt))
                    if receipt and receipt.get('steps',0)>0:
                        seconds = receipt.get('write_and_prediction_inclusive_seconds',receipt.get('full_bank_seconds'))
                        if seconds is not None and seconds>0:
                            rates.setdefault((variant,mode),[]).append(seconds/receipt['steps'])
    preflight = read(root/'preflight.json',{})
    fallback = {(r['variant'],r['mode']):r['info']['full_bank_seconds']/64
                for r in preflight.get('records',[]) if r['mode'] in ['wide','linear']}
    completed = sum(bool(e[2]) for e in entries)
    key = tuple(progress.get(k) for k in ['stage','role','variant','head_mode','fold'])
    remaining, estimated = 0., True
    for item,steps,receipt in entries:
        if receipt:
            continue
        pair = (item[2],item[3])
        rate = statistics.median(rates[pair]) if rates.get(pair) else fallback.get(pair)
        if rate is None:
            estimated = False
            continue
        elapsed = progress.get('seconds',0) if item == key else 0
        remaining += max(0,rate*steps-elapsed)
    speed = None
    if state == 'training' and progress.get('step',0)>0 and progress.get('seconds',0)>0:
        speed = progress['step']/progress['seconds']
    return dict(state=state,completed_banks=completed,total_banks=168,current=progress,
                current_label=LABELS.get(progress.get('variant')),steps_per_second=speed,
                estimated_remaining_seconds=remaining if estimated and state=='training' else None,
                final_steps_provisional=selection is None,error=job.get('error'),
                quality_claim='Точность цвета появится после выбора настроек и итоговой проверки; текущие значения — ход обучения.')
