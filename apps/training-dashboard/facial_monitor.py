"""Read-only progress for the new LaPa skin segmentation experiment; no ML imports."""
import json
import time
from pathlib import Path

from monitor import process_alive


def snapshot(root=Path('D:/Luma-RnD/data_growth_2026_09_14'),now=None,probe=process_alive):
    root = Path(root)
    now = time.time() if now is None else now

    def read(path,fallback=None):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            return fallback

    profile = read(root/'lapa/prepared_192/profile.json')
    if not profile:
        return None
    run = root/'facial_skin_v1'
    progress = read(run/'progress.json',{})
    protocol = read(run/'protocol.json',{})
    preflight = read(run/'preflight.json',{})
    job = read(run/'job.json',{})
    result = read(run/'test_results.json')
    history = read(run/'history.json',[])
    state = 'complete' if result else progress.get('status','ready')
    if state in ['training','evaluating']:
        live = probe(job.get('pid'))
        if live is not True:
            state = 'interrupted' if live is False else 'unknown'
        elif now-(run/'progress.json').stat().st_mtime > 150:
            state = 'stale'
    return dict(name='Luma ChromaSeed-Seg1',state=state,
                warming_up=state == 'training' and progress.get('step',0) < 20,
                parameters=preflight.get('parameters'),epochs=protocol.get('epochs'),
                train_images=profile['splits']['train']['rows'],validation_images=profile['splits']['val']['rows'],
                progress=progress,history=[dict(epoch=r['epoch'],iou=r['validation']['mean_image_iou'],loss=r['training_loss']) for r in history],
                test=result.get('test') if result else None,
                limitation='IoU оценивает маску кожи. Точность измерения оттенка этим тестом не проверяется.')
