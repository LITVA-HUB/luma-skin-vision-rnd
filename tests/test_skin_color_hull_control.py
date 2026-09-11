import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_color_hull_control import make_hull,project_hull


def test_tetrahedron_projection_and_kkt():
    palette=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    p,receipt=project_hull(np.array([[1.,1.,1.],[-1.,0.,0.],[.1,.2,.3]]),make_hull(palette))
    np.testing.assert_allclose(p,[[1/3,1/3,1/3],[0,0,0],[.1,.2,.3]],atol=1e-8)
    assert receipt['max_feasibility_violation']<1e-8 and receipt['max_relative_stationarity']<1e-8


def test_projection_leaves_observed_colors_fixed():
    rng=np.random.default_rng(17);palette=rng.normal(size=(30,3))
    out,_=project_hull(palette,make_hull(palette))
    np.testing.assert_array_equal(out,palette)
