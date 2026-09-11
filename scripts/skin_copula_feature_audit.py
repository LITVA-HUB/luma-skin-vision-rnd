"""Independent uint8 count-CDF ranks and scalar-patch descriptor checks."""
import json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_copula_data import load,OUT


def main():
    histograms=0;patches=0;gap=0.;transform={}
    for role in ['train','validation']:
        data=load(role)
        for i,rgb in enumerate(data['rgb']):
            image=rgb.reshape(-1,3);cdf=np.empty_like(image,dtype=np.float64)
            for channel in range(3):
                counts=np.bincount(image[:,channel],minlength=256)
                values=(np.cumsum(counts)-counts/2)/len(image)
                cdf[:,channel]=values[image[:,channel]]
            cells=np.floor(cdf*8).astype(int);counts=np.zeros(512,dtype=np.int64)
            np.add.at(counts,cells[:,0]*64+cells[:,1]*8+cells[:,2],1)
            np.testing.assert_array_equal((counts/len(image)).astype(np.float32),data['rank_hist'][i]);histograms+=1
            if i<(16 if role=='train' else 8):
                im=cdf.reshape(128,128,3)
                for row in range(8):
                    for col in range(8):
                        patch=im[row*16:(row+1)*16,col*16:(col+1)*16];p=patch.reshape(-1,3)
                        value=np.r_[np.quantile(p,[.1,.5,.9],axis=0).ravel(),p.mean(0),p.std(0),
                            .5*np.abs(np.diff(patch,axis=0)).mean((0,1))+.5*np.abs(np.diff(patch,axis=1)).mean((0,1))].astype(np.float32)
                        gap=max(gap,float(abs(value-data['rank_tokens'][i,row*8+col]).max()));patches+=1
    assert histograms==1230 and patches==1536 and gap<1e-7
    receipt=json.loads((OUT/'feature_receipt.json').read_bytes())
    for kind in ['strict_monotone','channel_mixing','clipping']:
        values=[r['rank_hist_total_variation'] for r in receipt['invariance_probe'] if r['derived_transform']==kind]
        transform[kind]={'cases':len(values),'mean_total_variation':float(np.mean(values)),'max_total_variation':max(values)}
    result={'status':'PASS','independent_count_cdf_histograms':histograms,'independent_patch_profiles':patches,
            'maximum_float32_patch_gap':gap,'derived_input_invariance_not_accuracy':transform,
            'feature_receipt_sha256':sha(OUT/'feature_receipt.json'),'audit_script_sha256':sha(Path(__file__))}
    (OUT/'feature_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))


if __name__=='__main__':main()
