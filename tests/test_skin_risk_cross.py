import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from skin_risk_cross import expected_error
from skin_mskcc_audit import scalar_de


def test_zero_variance_risk_is_actual_color_distance():
    pred=np.array([[40,14,20],[70,10,15]],float)
    center=np.array([[50,14,20],[70,10,15]],float)
    z=np.array([[-1,-1,-1],[1,1,1]],float)
    result=expected_error(pred,center,np.zeros_like(center),z)
    np.testing.assert_allclose(result,[scalar_de(a,b) for a,b in zip(pred,center)],atol=1e-12)


def test_risk_averages_each_image_separately():
    p=np.array([[50,10,20],[65,12,25]],float);s=np.array([[2,1,3],[4,3,2]],float)
    z=np.array([[-1,-1,-1],[1,1,1]],float)
    out=expected_error(p,p,s,z)
    expected=[np.mean([scalar_de(c,c+sigma*n) for n in z]) for c,sigma in zip(p,s)]
    np.testing.assert_allclose(out,expected,atol=1e-12)
