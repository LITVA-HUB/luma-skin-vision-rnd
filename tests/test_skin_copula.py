import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from skin_copula_data import rank_channels,joint_hist


def test_rank_context_exactly_invariant_to_strict_channel_maps_with_ties():
    x=np.random.default_rng(81).integers(0,256,(32,32,3))/255
    transformed=.03+.85*x**np.array([.6,1.4,2.2])
    np.testing.assert_array_equal(rank_channels(x),rank_channels(transformed))
    np.testing.assert_array_equal(joint_hist(rank_channels(x)),joint_hist(rank_channels(transformed)))


def test_ties_and_constant_channel_have_midrank():
    x=np.array([[0.,0.,1.],[0.,1.,1.],[1.,2.,1.],[2.,3.,1.]])
    y=rank_channels(x)
    np.testing.assert_array_equal(y[:,0],[.25,.25,.625,.875])
    np.testing.assert_array_equal(y[:,2],[.5,.5,.5,.5])


def test_histogram_matches_counts_and_preserves_mass():
    x=np.array([[0.,0.,0.],[1.,1.,1.],[.25,.5,.75],[.25,.5,.75]])
    h=joint_hist(x)
    assert h.shape==(512,) and h.sum()==1 and h[0]==.25 and h[-1]==.25 and h[2*64+4*8+6]==.5


def test_channel_mixing_is_not_claimed_invariant():
    x=np.random.default_rng(82).uniform(size=(64,64,3))
    matrix=np.array([[.8,.1,.1],[.15,.7,.15],[.1,.2,.7]])
    assert not np.array_equal(joint_hist(rank_channels(x)),joint_hist(rank_channels(x@matrix.T)))
