"""Audit achieved oracle codes; no claim of globally optimal infeasibility."""
import json
from pathlib import Path
import numpy as np
from skin_material_prior import PRIOR,material_value_jacobian
from skin_material_train import OUT,RUN
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de
from skin_pair_train import write


def main():
    result=json.loads((OUT/'feasibility_train.json').read_bytes())
    for p,h in result['bindings'].items():assert sha(ROOT/p)==h,p
    prior=dict(np.load(PRIOR));tr=load('train')
    targets=np.stack([tr['target'][tr['site']==site][0] for site in sorted(set(tr['site']))])
    arrays=np.load(RUN/'feasibility_train.npz');gap=0.;cases=0
    for r in result['records']:
        k=int(r['coefficient_bound']);code=arrays[f'b{k}_code'];p=arrays[f'b{k}_prediction']
        assert np.max(np.abs(code))<=k+1e-12
        replay=np.stack([material_value_jacobian(prior['mu']+prior['basis']@z,prior['basis'],prior['matrix'],prior['white'])[0] for z in code])
        np.testing.assert_array_equal(replay,p)
        e=np.array([scalar_de(a,b) for a,b in zip(p,targets,strict=True)])
        gap=max(gap,float(np.max(np.abs(e-arrays[f'b{k}_error']))),abs(float(e.mean())-r['mean_achieved_delta_e00']))
        assert float(np.mean(e<.01))==r['fraction_below_0_01']
        cases+=len(e)
    assert gap<1e-10
    audit={'status':'PASS','independent_scalar_cases':cases,'maximum_metric_gap':gap,
        'achieved_codes_replay_exactly':True,'coefficient_boxes_checked':True,
        'global_optimality_or_solver_status_replayed':False,'validation_or_test_targets_fit':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'feasibility_train.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'feasibility_audit.json',audit);print(json.dumps(audit))


if __name__=='__main__':main()
