"""Unchanged six local-reference systems on explicit source camera banks."""
import argparse,json
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_local_reference import METHODS,predict_bank
from skin_local_reference_run import summarize,OUT as SCREEN
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import write,subset

OUT=ROOT/'docs/benchmarks/skin_local_reference_transfer_v1';RUN=ROOT/'experiments/runs/skin_local_reference_transfer_v1'
PROTOCOL=ROOT/'docs/research/skin_local_reference_transfer_protocol_v1.md'


def bindings():
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/skin_local_reference.py',ROOT/'scripts/skin_local_reference_run.py',
           ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'scripts/skin_mskcc_data.py',ROOT/'src/luma_skin_vision/color.py',
           SCREEN/'source_lock.json',SCREEN/'results.json',SCREEN/'audit.json',
           ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz',ROOT/'data/processed/skin_mskcc_pixels_v1/validation.npz']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def banks(tr,va):
    assert set(tr['patient']).isdisjoint(va['patient']) and set(tr['site']).isdisjoint(va['site'])
    yield 'mixed',tr,{'known':np.ones(len(va['target']),bool)}
    for device in ('SLR','ipod'):
        t=subset(tr,tr['device']==device);mask=va['device']==device
        assert set(t['device']).isdisjoint(va['device'][~mask])
        yield 'from_'+device,t,{'known':mask,'unseen':~mask}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    assert json.loads((SCREEN/'audit.json').read_bytes())['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'source_lock.json';bound=bindings()
    if args.stage=='prepare':
        if lock.exists():assert json.loads(lock.read_bytes())['bindings']==bound
        else:write(lock,{'bindings':bound,'systems':METHODS,'banks':3,'method_domain_evaluations':30})
        print('Frozen source transfer follow-up');return
    assert json.loads(lock.read_bytes())['bindings']==bound
    if (OUT/'results.json').exists():raise ValueError('Preserve existing results')
    RUN.mkdir(parents=True,exist_ok=True);tr,va=load('train'),load('validation');records=[]
    for protocol,t,domains in banks(tr,va):
        got=predict_bank(t['color'],t['target'],va['color']);pred=got.pop('predictions');risk=got['risk']
        np.savez(RUN/(protocol+'.npz'),**pred,**got,target=va['target'],patient=va['patient'],site=va['site'])
        for domain,mask in domains.items():
            ev=subset(va,mask);rr=risk[mask];order=np.argsort(rr,kind='stable');base=delta_e00(pred['global_ridge'][mask],ev['target'])
            for method in METHODS:
                p=pred[method][mask];e=delta_e00(p,ev['target']);curve=np.cumsum(e[order])/np.arange(1,len(e)+1);coverage=[]
                for c in (1.,.95,.9,.8,.7,.6):
                    n=int(np.ceil(c*len(e)));ix=order[:n];coverage.append({'coverage':c,'accepted':n,**summarize(e[ix],ev['patient'][ix],ev['site'][ix])})
                name=f'{protocol}__{domain}__{method}';file=RUN/(name+'.npz');np.savez(file,prediction=p,error=e,risk=rr,curve=curve,order=order)
                pd=np.array([(e-base)[ev['patient']==a].mean() for a in np.unique(ev['patient'])])
                row={'name':name,'protocol':protocol,'domain':domain,'method':method,'train_people':len(set(t['patient'])),'train_images':len(t['target']),
                    'train_cameras':np.unique(t['device']).tolist(),'evaluation_cameras':np.unique(ev['device']).tolist(),'people':len(set(ev['patient'])),'images':len(e),
                    'full':summarize(e,ev['patient'],ev['site']),'coverage':coverage,'mean_delta_vs_ridge':float((e-base).mean()),'improved_people_vs_ridge':int((pd<0).sum()),
                    'array_sha256':sha(file),'uniform_limit_max_gap':got['uniform_limit_max_gap']}
                records.append(row);print(json.dumps({'case':name,'mean':row['full']['mean'],'at80':coverage[3]['mean']}),flush=True)
    assert bindings()==bound
    write(OUT/'results.json',{'results':records,'source_exploratory_only':True,'independent_accuracy_changed':False,
          'source_lock_sha256':sha(lock),'arrays':{p.name:sha(p) for p in RUN.glob('*.npz')}})


if __name__=='__main__':main()
