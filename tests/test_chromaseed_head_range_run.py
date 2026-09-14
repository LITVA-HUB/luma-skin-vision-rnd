import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_pair_control_is_read_only_parent_and_unknown_heads_rejected():
    from chromaseed_architecture_scale_run import bank_path as parent_path
    from chromaseed_head_range_run import RUN, bank_path, pair_key
    assert bank_path('mixed','patch5m','unit',0)==parent_path('mixed','patch5m',0)
    assert bank_path('mixed','patch5m','wide',0).is_relative_to(RUN)
    with pytest.raises(ValueError):
        pair_key('patch5m','unregistered')


def test_policy_can_reject_all_new_heads_and_keeps_per_pair_results():
    from chromaseed_head_range_run import MODES, VARIANTS, pair_key, policies
    rows=[dict(variant=pair_key(v,m),architecture=v,head_mode=m,clean=1. if m=='unit' else 2.,
               p90=3.,numeric_bytes=100,step=128,lr=1e-5) for v in VARIANTS for m in MODES]
    control=dict(variant='np',clean=.5,p90=1.,numeric_bytes=50,step=0,lr=None)
    result=policies(rows+[control])
    assert result['overall']['variant']=='np'
    assert len(result['per_pair'])==21 and len(result['per_architecture'])==7
    assert all(c['head_mode']=='unit' for c in result['per_architecture'].values())
