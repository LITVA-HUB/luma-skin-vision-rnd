"""Independent fit-only grid, scalar color costs and full Gram identity audit."""
import json
from pathlib import Path
import numpy as np
from skin_loss_field_train import OUT,PRIOR
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import subset,write


def main():
    tr=load('train');cases=0;max_scalar=0.;max_gram=0.;max_loss=0.;bindings={}
    for protocol,t in [('mixed',tr)]+[('from_'+c,subset(tr,tr['device']==c)) for c in ('SLR','ipod')]:
        path=PRIOR/f'{protocol}.npz';a=dict(np.load(path));bindings[str(path.relative_to(ROOT))]=sha(path)
        atoms,index=np.unique(t['target'],axis=0,return_inverse=True)
        np.testing.assert_array_equal(a['atoms'],atoms);np.testing.assert_array_equal(a['index'],index)
        for j in range(3):np.testing.assert_array_equal(np.unique(a['grid'][:,j]),np.linspace(atoms[:,j].min()-5,atoms[:,j].max()+5,25))
        rng=np.random.default_rng(11413);js=rng.choice(len(a['grid']),128,replace=False)
        for i in range(len(atoms)):
            for j in js:
                expected=scalar_de(atoms[i],a['grid'][j]);max_scalar=max(max_scalar,abs(expected-float(a['cost'][i,j])));cases+=1
        c=a['cost'].astype(float);c-=c.mean(0)
        gram=c@c.T/c.shape[1]/float(a['scale'])**2
        actual=a['phi'].astype(float)@a['phi'].astype(float).T
        max_gram=max(max_gram,float(np.max(abs(actual-gram))))
        weights=rng.dirichlet(np.ones(len(atoms)),size=32)
        difference=weights-np.eye(len(atoms))[rng.integers(0,len(atoms),32)]
        x=np.mean((difference@a['cost'])**2,axis=1)/float(a['scale'])**2
        y=np.sum((difference@a['phi'])**2,axis=1);max_loss=max(max_loss,float(np.max(abs(x-y))))
    assert max_scalar<4e-6 and max_gram<2e-6 and max_loss<2e-6
    bindings.update({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/skin_mskcc_audit.py',OUT/'palette_receipt.json']})
    record={'status':'PASS','scalar_cost_cases':cases,'max_float32_cost_gap':max_scalar,'max_gram_gap':max_gram,
        'max_candidate_mse_identity_gap':max_loss,'full_rank_no_truncation':True,'train_only_dictionary_grid':True,
        'validation_or_test_loaded':False,'bindings':bindings}
    write(OUT/'palette_audit.json',record);print(json.dumps(record))


if __name__=='__main__':main()
