"""Explicit empirical color risk and calibration-threshold evaluation."""
import numpy as np
from skin_mskcc_summary_pilot import summarize

COVERAGES=[1,.95,.9,.8,.7,.6]


def rows(data):
    return [{k:data[k][i].item() for k in ['image','patient','site','device','image_type']} for i in range(len(data['target']))]


def curve(error,score,data):
    order=np.lexsort((data['image'],score));risk=np.cumsum(error[order])/np.arange(1,len(error)+1)
    return order,risk


def selection(error,score,data):
    _,risk=curve(error,score,data);n=len(error)
    return float(risk[int(np.ceil(.8*n))-1]),float(risk[int(np.ceil(.6*n))-1:].mean())


def evaluate(error,score,data):
    r=rows(data);order,risk=curve(error,score,data)
    result={'full':summarize(error,r),'coverage':[],'strata':{},'full_curve_mean':risk.tolist()}
    for c in COVERAGES:
        ix=order[:int(np.ceil(c*len(error)))];result['coverage'].append({'coverage':c,**summarize(error[ix],[r[i] for i in ix])})
    for key in ['device','image_type']:
        result['strata'][key]={}
        for value in np.unique(data[key]):
            ix=np.flatnonzero(data[key]==value);result['strata'][key][str(value)]=summarize(error[ix],[r[i] for i in ix])
    return result


def accepted(error,mask,data):
    ix=np.flatnonzero(mask);result={'coverage':len(ix)/len(error),'accepted':len(ix)}
    if len(ix):result.update(summarize(error[ix],[rows(data)[i] for i in ix]))
    else:result['mean']=None
    return result
