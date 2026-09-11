"""TRAIN-only evidence for using actual same-site patch observations."""
import itertools,json
from pathlib import Path
import numpy as np
from skin_mskcc_pixels import load
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write

OUT=ROOT/'docs/benchmarks/skin_capture_support_v1'


def main():
    t=load('train');pairs=[];coordinates=[];zero=0;mode_counts={};camera_cross=0;targetgap=0.;site_camera=[]
    token=t['tokens'].astype(float);pmean=token[:,:,9:12]
    for site in np.unique(t['site']):
        indices=np.flatnonzero(t['site']==site);site_camera.append(len(set(t['device'][indices])))
        assert len(set(t['patient'][indices]))==1
        for a,b in itertools.combinations(indices,2):
            targetgap=max(targetgap,float(np.max(abs(t['target'][a]-t['target'][b]))))
            camera_cross+=int(t['device'][a]!=t['device'][b]);pairs.append((a,b))
            key=' / '.join(sorted([str(t['mode'][a]),str(t['mode'][b])]))
            mode_counts[key]=mode_counts.get(key,0)+1
            # A coordinate-index pattern diagnostic, not a registration test.
            x=pmean[a]-pmean[a].mean(0);y=pmean[b]-pmean[b].mean(0)
            den=np.linalg.norm(x)*np.linalg.norm(y)
            if den<1e-12:zero+=1
            else:coordinates.append(float(np.sum(x*y)/den))
    assert targetgap==0
    pair=np.asarray(pairs,dtype=int);a,b=pair.T
    delta=np.linalg.norm(pmean[a].mean(1)-pmean[b].mean(1),axis=1)
    site_sizes=[int(np.sum(t['site']==s)) for s in np.unique(t['site'])]
    def stats(x):return {'mean':float(np.mean(x)),'median':float(np.median(x)),'p05':float(np.quantile(x,.05)),'p95':float(np.quantile(x,.95)),'max':float(np.max(x))}
    # Independent loop checks against vectorized mean-RGB shifts.
    check=[]
    for i in np.linspace(0,len(pair)-1,64,dtype=int):
        ai,bi=pair[i];left=np.array([sum(float(v) for v in pmean[ai,:,c])/64 for c in range(3)])
        right=np.array([sum(float(v) for v in pmean[bi,:,c])/64 for c in range(3)])
        check.append(abs(float(np.linalg.norm(left-right))-delta[i]))
    assert max(check)<1e-12
    OUT.mkdir(parents=True,exist_ok=True)
    result={'scope':'TRAIN only; paired appearance diagnostics, no new image model results',
        'images':len(t['target']),'people':len(set(t['patient'])),'sites':len(site_sizes),'unordered_pairs':len(pair),
        'site_size_counts':{str(n):site_sizes.count(n) for n in sorted(set(site_sizes))},
        'cross_camera_pairs':camera_cross,'multi_camera_sites':sum(n>1 for n in site_camera),
        'maximum_same_site_native_Lab_gap':targetgap,'mode_pair_counts':mode_counts,
        'within_pair_mean_RGB_shift_0_1':stats(delta),'coordinate_pattern_cosine':stats(coordinates),
        'coordinate_degenerate_pairs':zero,'coordinate_cosine_below_0_5_fraction':float(np.mean(np.array(coordinates)<.5)),
        'registration_or_pure_camera_effect_proven':False,'max_independent_mean_shift_gap':max(check),
        'decision':'Use observed patch-bag resampling only; no registered pixel interpolation or cross-camera pair claim',
        'validation_or_test_loaded':False,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'scripts/skin_mskcc_pixels.py',ROOT/'data/processed/skin_mskcc_pixels_v1/train.npz']}}
    write(OUT/'train_pair_audit.json',result);print(json.dumps(result))


if __name__=='__main__':main()
