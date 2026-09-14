import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent))
from facial_monitor import snapshot


def test_facial_monitor_reads_actual_data_and_marks_dead_job(tmp_path):
    p=tmp_path/'lapa/prepared_192/profile.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(dict(splits=dict(train=dict(rows=15914),val=dict(rows=1692)))))
    run=tmp_path/'facial_skin_v1'
    run.mkdir()
    (run/'progress.json').write_text(json.dumps(dict(status='training',epoch=2)))
    (run/'job.json').write_text(json.dumps(dict(pid=123)))
    row=snapshot(tmp_path,probe=lambda pid:False)
    assert row['train_images']==15914 and row['state']=='interrupted'
    row=snapshot(tmp_path,probe=lambda pid:True)
    assert row['state']=='training'
    assert snapshot(tmp_path,now=10**12,probe=lambda pid:True)['state']=='stale'


def test_facial_monitor_missing_data_is_not_mocked(tmp_path):
    assert snapshot(tmp_path) is None
