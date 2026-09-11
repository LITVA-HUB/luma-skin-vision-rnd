"""Post-hoc sensitivity to duplicate spectra; no refitting or split rewriting."""
from collections import defaultdict
import hashlib
import json
import numpy as np
from skin_issa_data import OUT, PRIVATE, ROOT, sha, write_json
from skin_issa_material import load_cache, verify_lock, stats


def main():
    verify_lock()
    tr,common,x,_=load_cache('train');va,_,y,valid=load_cache('validation')
    groups=defaultdict(set)
    for data,array in [(tr,x),(va,y)]:
        for row,subject in zip(array,data['subject'],strict=True):
            groups[hashlib.sha256(np.asarray(row,dtype='<f8').tobytes()).hexdigest()].add(str(subject))
    graph=defaultdict(set)
    for subjects in groups.values():
        for s in subjects:graph[s].update(subjects-{s})
    reached=set(tr['subject']);queue=list(reached)
    while queue:
        s=queue.pop()
        for other in graph[s]-reached:reached.add(other);queue.append(other)
    excluded=set(va['subject'])&reached
    # Independent fixed-point propagation over fingerprints checks the graph.
    other=set(tr['subject'])
    while True:
        updated=set(other)
        for group in groups.values():
            if group&other:updated.update(group)
        if updated==other:break
        other=updated
    assert excluded==set(va['subject'])&other
    color=np.all(va['declared_support'][valid]==common[None,:],axis=1)
    cp=va['subject'][valid][color];co=va['origin'][valid][color]
    keep=np.array([p not in excluded for p in cp]);spectral_keep=np.array([p not in excluded for p in va['subject'][valid]])
    results=json.loads((OUT/'material_results.json').read_text(encoding='utf-8-sig'))
    arrays=np.load(PRIVATE/'material_validation_predictions.npz',allow_pickle=False)
    rows=[]
    for r in results['records']:
        name=f"{r['method']}_{r['width']}"
        e=arrays[name+'_error'][keep]
        spectral_error=arrays[name+'_spectra'][spectral_keep]-y[valid][spectral_keep]
        rmse=np.sqrt(np.mean(spectral_error**2,axis=1))
        relative=np.linalg.norm(spectral_error,axis=1)/np.linalg.norm(y[valid][spectral_keep],axis=1)
        by_source={str(o):stats(e[co[keep]==o]) for o in set(co[keep])}
        rows.append({'method':r['method'],'width':r['width'],'color_records':int(keep.sum()),
            'spectral_records':int(spectral_keep.sum()),'spectral_rmse':stats(rmse),'spectral_relative_l2':stats(relative),
            'delta_e00':stats(e),'mean_subject_delta_e00':float(np.mean([e[cp[keep]==p].mean() for p in set(cp[keep])])),
            'worst_source_mean':max(v['mean'] for v in by_source.values())})
    report={'scope':'POST-HOC descriptive exclusion; original models/split/results unchanged; not an independent confirmation',
        'exclusion_rule':'Remove every validation subject label connected to any TRAIN label through any exact common31 spectrum, including transitive connections',
        'excluded_validation_subject_labels':len(excluded),
        'excluded_validation_records':int((~spectral_keep).sum()),
        'excluded_color_records':int((~keep).sum()),
        'remaining_color_records':int(keep.sum()),
        'excluded_labels_by_origin':{str(o):len(set(va['subject'][va['origin']==o])&excluded) for o in set(va['origin'])},
        'graph_and_fixed_point_agree':True,'records':rows,
        'limitation':'Exact spectra can indicate duplicated records, not necessarily identical people; remaining near-duplicates or differently labelled people are unresolved.',
        'no_refit_or_hyperparameter_selection':True,'reserved_numerical_endpoints_read':False,
        'bindings':{p:sha(ROOT/p) for p in ['scripts/skin_issa_overlap.py',
            'docs/benchmarks/skin_issa_v1/material_results.json','docs/benchmarks/skin_issa_v1/independent_audit.json']}}
    write_json(OUT/'overlap_sensitivity.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'records','bindings'}}))
    print(json.dumps([r for r in rows if (r['method'],r['width']) in [('reflectance',3),('density',8)]]))


if __name__=='__main__':main()
