import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from cc_projector_probe import cone_oracle, projector_features


def test_projector_is_basis_independent_and_refuses_low_rank():
    generator=torch.Generator().manual_seed(10)
    x=torch.rand(2,64,3,generator=generator,dtype=torch.float64)
    probes=torch.randn(64,16,generator=generator,dtype=torch.float64)
    matrix=torch.tensor([[1.2,.2,.1],[.1,.8,.3],[.2,.1,1.4]],dtype=torch.float64)
    a=projector_features(x,probes)
    b=projector_features(x@matrix.T,probes)
    assert a["valid"].all() and b["valid"].all()
    torch.testing.assert_close(a["features"],b["features"],atol=1e-12,rtol=1e-12)
    flat=projector_features(torch.ones_like(x),probes)
    assert not flat["valid"].any()
    assert torch.isfinite(flat["features"]).all()


def test_cone_oracle_hits_feasible_target_and_certifies_projection():
    x=np.array([[1.,.1,.1],[.1,1.,.1],[.1,.1,1.]])
    gt=np.array([.3,.7,.2])@x
    result=cone_oracle(x,gt)
    assert result["recovery"]<1e-5 and result["kkt_violation"]<1e-10
    # Rank-one cone: only one possible chromaticity, checked independently.
    x=np.array([[1.,2.,3.],[2.,4.,6.]])
    gt=np.ones(3)
    result=cone_oracle(x,gt)
    cosine=np.dot(x[0],gt)/(np.linalg.norm(x[0])*np.linalg.norm(gt))
    expected=np.degrees(np.arccos(cosine))
    assert abs(result["recovery"]-expected)<1e-10
    assert result["kkt_violation"]<1e-10
