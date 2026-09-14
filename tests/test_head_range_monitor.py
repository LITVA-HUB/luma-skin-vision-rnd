import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apps/training-dashboard'))


def put(root,name,value):
    path=root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_dead_worker_cannot_show_live_speed_or_eta(tmp_path):
    from head_range_monitor import snapshot
    put(tmp_path,'source_lock.json',{})
    put(tmp_path,'job.json',dict(status='running',pid=123))
    put(tmp_path,'progress.json',dict(status='training',step=128,seconds=10))
    s=snapshot(tmp_path,probe=lambda pid:False)
    assert s['state']=='interrupted' and s['steps_per_second'] is None
    assert s['estimated_remaining_seconds'] is None


def test_forecast_counts_each_registered_architecture_and_respects_failure(tmp_path):
    from head_range_monitor import snapshot
    from monitor import VARIANTS
    put(tmp_path,'source_lock.json',{})
    put(tmp_path,'job.json',dict(status='running',pid=123))
    put(tmp_path,'progress.json',dict(status='training',stage='inner',role='mixed',variant='patch_small',head_mode='wide',fold=0,step=128,seconds=10))
    put(tmp_path,'preflight.json',dict(records=[dict(variant=v,mode=m,info=dict(full_bank_seconds=64)) for v in VARIANTS for m in ['wide','linear']]))
    s=snapshot(tmp_path,probe=lambda pid:True)
    assert s['estimated_remaining_seconds']==168*2048-10
    assert s['steps_per_second']==12.8 and s['completed_banks']==0
    put(tmp_path,'job.json',dict(status='failed',pid=123,error='Recorded failure'))
    assert snapshot(tmp_path,probe=lambda pid:True)['state']=='failed'
