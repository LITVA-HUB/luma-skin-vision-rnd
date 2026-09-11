"""Repeatable structural audit of only the licensed, preassigned TRAIN cube."""
import json
import numpy as np
from scipy.io import loadmat
from skin_mskcc_data import sha
from skin_uminho_acquire import RAW,OUT


def main():
    receipt=json.loads((OUT/'acquisition.json').read_bytes());manifest=RAW/'manifest.json'
    assert sha(manifest)==receipt['manifest_sha256']
    rows=json.loads(manifest.read_bytes())['rows']
    selected=min((r for r in rows if r['role']=='train'),key=lambda r:r['size'])
    path=(RAW/selected['name']).resolve()
    assert path.is_relative_to(RAW.resolve()) and sha(path)==receipt['pilot_cube_sha256']
    a=loadmat(path)['datao'];assert a.ndim==3 and a.shape[2]==33
    mask=np.any(a!=0,axis=2);s=a[mask]
    result={'scope':'One preassigned TRAIN cube only; no rendering or model score','shape':list(a.shape),'dtype':str(a.dtype),
        'nonzero_pixels':int(mask.sum()),'background_zero_pixels':int((~mask).sum()),'finite_fraction':float(np.isfinite(a).mean()),
        'minimum_nonbackground_reflectance':float(s.min()),'maximum_nonbackground_reflectance':float(s.max()),
        'negative_values':int((s<0).sum()),'values_above_one':int((s>1).sum()),
        'reflectance_quantiles':np.quantile(s,[0,.01,.5,.99,1]).tolist(),'skin_foreground_mask_is_verified':False}
    target=OUT/'train_cube_audit.json'
    if target.exists():assert json.loads(target.read_bytes())==result
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
