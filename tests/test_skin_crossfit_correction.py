"""Subject roles, matched inclusion controls and stable color-coordinate heads."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_inner_folds_exclude_whole_people_and_match_camera_support():
    from skin_crossfit_correction import inner_folds
    people=np.repeat(np.arange(18),3);camera=np.where(people<6,'SLR','ipod')
    folds=inner_folds(people,camera)
    assert set(folds)=={0,1,2}
    for f in range(3):
        held=set(people[folds==f]);train=set(people[folds!=f])
        assert len(held)==6 and len(train)==12 and held.isdisjoint(train)
        assert len(set(people[(folds==f)&(camera=='SLR')]))==2
        assert len(set(people[(folds==f)&(camera=='ipod')]))==4
    perm=np.random.default_rng(9).permutation(len(people))
    np.testing.assert_array_equal(inner_folds(people[perm],camera[perm]),folds[perm])
    for p in np.unique(people):assert len(np.unique(folds[people==p]))==1
    with pytest.raises(ValueError):inner_folds(people[:-6],camera[:-6])


def test_crossfit_and_same_size_in_sample_use_distinct_correct_encoders():
    from skin_crossfit_correction import choose_predictions
    fold=np.array([0,1,2,0,2]);pred=np.stack([np.full((len(fold),3),k,dtype=float) for k in range(3)])
    oof,included=choose_predictions(pred,fold)
    np.testing.assert_array_equal(oof[:,0],fold)
    np.testing.assert_array_equal(included[:,0],(fold+1)%3)
    changed=pred.copy()
    # Changes to predictions from encoders that SAW a person cannot change OOF.
    for i,f in enumerate(fold):changed[np.arange(3)!=f,i]+=10000
    np.testing.assert_array_equal(choose_predictions(changed,fold)[0],oof)


def test_stable_head_coordinates_capacity_and_zero_initial_correction():
    from skin_crossfit_correction import StableHead,features
    rng=np.random.default_rng(79);color=rng.normal(size=(7,36));lab=rng.normal(size=(7,3))+[50,10,15]
    cm=color.mean(0);cs=np.maximum(color.std(0),1e-6);ym=lab.mean(0);ys=np.maximum(lab.std(0),1e-6)
    x=features(color,lab,cm,cs,ym,ys)
    np.testing.assert_allclose(x[:,:36]*cs+cm,color,atol=1e-6)
    np.testing.assert_allclose(x[:,36:]*ys+ym,lab,atol=1e-6)
    torch.manual_seed(6);head=StableHead();assert sum(p.numel() for p in head.parameters())==188035
    assert 929297+188035==1117332<=1129297
    out=head(torch.from_numpy(x));assert torch.equal(out,torch.zeros(7,3))
    target=torch.randn(7,3);loss=(out-target).square().mean();loss.backward()
    assert torch.isfinite(head.net[-2].weight.grad).all() and head.net[-2].weight.grad.abs().sum()>0
