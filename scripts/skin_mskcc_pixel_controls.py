"""Nine predeclared local-pixel controls; source validation only."""
import json
import warnings
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from threadpoolctl import threadpool_limits
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load,PROTOCOL
from skin_mskcc_summary_pilot import summarize


def main():
    train,val=load('train'),load('validation')
    out=ROOT/'docs/benchmarks/skin_mskcc_pixels_v1';run=ROOT/'experiments/runs/skin_mskcc_pixel_controls_v1'
    run.mkdir(exist_ok=False,parents=True)
    rows=[{k:val[k][i].item() for k in ['image','patient','site','device','image_type']} for i in range(len(val['target']))]
    results={'protocol_sha256':sha(PROTOCOL),'source_validation_only':True,'models':[]}
    for feature in ['median','color','hist','color_hist']:
        x,vx=train[feature],val[feature]
        scaling=StandardScaler().fit(x)
        risk=NearestNeighbors(n_neighbors=5).fit(scaling.transform(x)).kneighbors(scaling.transform(vx))[0].mean(1)
        candidates=[('ridge',make_pipeline(StandardScaler(),Ridge(alpha=1))),
                    ('mlp',TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64,32),activation='tanh',solver='lbfgs',alpha=1,max_iter=2000,random_state=17)),transformer=StandardScaler()))]
        if feature=='median':
            candidates.append(('poly2',make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),Ridge(alpha=1))))
        for label,model in candidates:
            name=feature+'_'+label
            with warnings.catch_warnings(record=True) as caught,threadpool_limits(4):
                warnings.simplefilter('always');model.fit(x,train['target'])
            pred=model.predict(vx);e=delta_e00(pred,val['target'])
            path=run/(name+'.joblib');joblib.dump(model,path)
            np.savez(run/(name+'.npz'),prediction=pred,target=val['target'],risk=risk,delta_e00=e)
            if not np.array_equal(joblib.load(path).predict(vx),pred):raise ValueError('Replay mismatch')
            record={'name':name,'full':summarize(e,rows),'model_sha256':sha(path),
                    'warnings':[str(w.message) for w in caught], 'strata':{}}
            order=np.lexsort((val['image'],risk));record['coverage']=[]
            for coverage in [1,.95,.9,.8,.7,.6]:
                ix=order[:int(np.ceil(coverage*len(e)))];record['coverage'].append({'coverage':coverage,**summarize(e[ix],[rows[i] for i in ix])})
            for key in ['device','image_type']:
                record['strata'][key]={}
                for value in np.unique(val[key]):
                    ix=np.flatnonzero(val[key]==value);record['strata'][key][str(value)]=summarize(e[ix],[rows[i] for i in ix])
            results['models'].append(record)
            (out/'controls.partial.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
            print(json.dumps({'control':name,'mean_delta_e00':record['full']['mean'],'median':record['full']['median'],'p95':record['full']['p95'],'warnings':record['warnings']}),flush=True)
    results['selected']=min(results['models'],key=lambda r:r['full']['patient_balanced_mean'])['name']
    (out/'controls.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
