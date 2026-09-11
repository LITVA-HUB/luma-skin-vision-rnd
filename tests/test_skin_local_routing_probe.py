import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_local_routing_probe import route


def test_local_and_global_routing_agree_for_constant_patch_gate():
    r=np.random.default_rng(31);h=r.normal(size=(3,5,4,3));w=r.dirichlet(np.ones(5),size=3);g=r.dirichlet(np.ones(4),size=3)
    np.testing.assert_allclose(route(h,w,g),route(h,w,np.repeat(g[:,None],5,axis=1)),atol=1e-14)


def test_varying_gate_need_not_commute_with_pooling():
    h=np.array([[[[50.,10.,20.],[70.,10.,20.]],[[70.,10.,20.],[50.,10.,20.]]]])
    w=np.array([[.5,.5]]);local=np.array([[[1.,0.],[0.,1.]]])
    np.testing.assert_allclose(route(h,w,local),[[50,10,20]])
    np.testing.assert_allclose(route(h,w,local.mean(1)),[[60,10,20]])
