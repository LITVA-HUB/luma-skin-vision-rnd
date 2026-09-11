"""Independent source-bank exclusion, weighted solves and color-risk replay."""
import json,math
from pathlib import Path
import numpy as np
from skin_local_reference_transfer import OUT,RUN,bindings
from skin_local_reference_verify import verify_bank,verify_metrics
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_mskcc_audit import scalar_de
from skin_pair_train import write


def main():
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    r=json.loads((OUT/'results.json').read_bytes());tr,va=load('train'),load('validation')
    for name,h in r['arrays'].items():assert sha(RUN/name)==h
    assert set(tr['patient']).isdisjoint(va['patient']) and set(tr['site']).isdisjoint(va['site'])
    count=0;gap=0.;wgap=0.;ugap=0.;scalar_gap=0.;coverage=0;arrays=0;solves=0
    methods=('global_ridge','global_mean','appearance_mean','appearance_affine','color_mean','color_affine')
    for protocol,camera in [('mixed',None),('from_SLR','SLR'),('from_ipod','ipod')]:
        mask=np.ones(len(tr['target']),bool) if camera is None else tr['device']==camera
        s=dict(np.load(RUN/(protocol+'.npz')))
        for key in ('target','patient','site'):np.testing.assert_array_equal(s[key],va[key])
        g,w,c,u=verify_bank(tr['color'][mask],tr['target'][mask],va['color'],s,{name:s[name] for name in methods})
        gap=max(gap,g);wgap=max(wgap,w);ugap=max(ugap,u);count+=c;solves+=2*len(va['target'])
        for domain in (('known',) if camera is None else ('known','unseen')):
            ev=np.ones(len(va['target']),bool) if camera is None else va['device']==camera
            if domain=='unseen':ev=~ev;assert set(tr['device'][mask]).isdisjoint(va['device'][ev])
            order=np.argsort(s['risk'][ev],kind='stable');baseline=np.array([scalar_de(a,b) for a,b in zip(s['global_ridge'][ev],va['target'][ev])]);count+=len(baseline)
            for method in methods:
                name=f'{protocol}__{domain}__{method}';m=next(a for a in r['results'] if a['name']==name);z=dict(np.load(RUN/(name+'.npz')))
                assert m['train_images']==mask.sum() and m['train_people']==len(set(tr['patient'][mask]))
                assert m['images']==ev.sum() and m['people']==len(set(va['patient'][ev]))
                assert m['train_cameras']==np.unique(tr['device'][mask]).tolist() and m['evaluation_cameras']==np.unique(va['device'][ev]).tolist()
                np.testing.assert_array_equal(z['prediction'],s[method][ev]);np.testing.assert_array_equal(z['risk'],s['risk'][ev]);np.testing.assert_array_equal(z['order'],order);arrays+=1
                e=np.array([scalar_de(a,b) for a,b in zip(z['prediction'],va['target'][ev])]);count+=len(e)
                scalar_gap=max(scalar_gap,float(abs(e-z['error']).max()));verify_metrics(e,va['patient'][ev],va['site'][ev],m['full'])
                curve=np.array([math.fsum(e[order[:i]])/i for i in range(1,len(e)+1)])
                np.testing.assert_allclose(curve,z['curve'],atol=1e-11,rtol=1e-12)
                for c,row in zip((1.,.95,.9,.8,.7,.6),m['coverage']):
                    n=math.ceil(c*len(e));ix=order[:n];assert row['coverage']==c and row['accepted']==n
                    verify_metrics(e[ix],va['patient'][ev][ix],va['site'][ev][ix],row);coverage+=1
                pd=np.array([(e-baseline)[va['patient'][ev]==a].mean() for a in np.unique(va['patient'][ev])])
                assert int((pd<0).sum())==m['improved_people_vs_ridge'] and abs((e-baseline).mean()-m['mean_delta_vs_ridge'])<1e-10
        print(json.dumps({'verified_bank':protocol,'scalar_cases':count}),flush=True)
    assert arrays==30 and coverage==180 and solves==1584 and scalar_gap<1e-10
    assert json.loads((OUT/'source_lock.json').read_bytes())['bindings']==bindings()
    audit={'status':'PASS','source_banks':3,'weighted_augmented_solves':solves,'uniform_limit_checks':792,'exact_domain_arrays':arrays,
        'scalar_color_cases':count,'coverage_rows':coverage,'max_scalar_gap':scalar_gap,'max_local_prediction_gap':gap,
        'max_weight_gap':wgap,'max_uniform_prediction_gap':ugap,'source_exploratory_only':True,'independent_accuracy_changed':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/skin_local_reference_verify.py',OUT/'results.json',OUT/'source_lock.json',ROOT/'scripts/skin_mskcc_audit.py']}}
    write(OUT/'audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
