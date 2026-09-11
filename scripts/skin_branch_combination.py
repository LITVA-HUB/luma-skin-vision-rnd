"""Fixed equal-weight combinations; no blend fitting, no new endpoint access."""
import json
from pathlib import Path
import numpy as np
from luma_skin_vision.color import delta_e00
from skin_mskcc_data import ROOT,sha
from skin_mskcc_audit import scalar_de

OUT=ROOT/'docs/benchmarks/skin_branch_combination_v1'
PROTOCOL=ROOT/'docs/research/skin_branch_combination_protocol_v1.md'
PAIRS={
 'mixture_graph_always':(('skin_capture_v1','mixture_mse'),('skin_train_branch_v1','graph_always')),
 'mixture_graph_drop':(('skin_capture_v1','mixture_mse'),('skin_train_branch_v1','graph_drop')),
 'mixture_conv_drop':(('skin_capture_v1','mixture_mse'),('skin_train_branch_v1','conv_drop')),
 'mixture_plain':(('skin_capture_v1','mixture_mse'),('skin_capture_v1','plain_mse')),
 'plain_graph_always':(('skin_capture_v1','plain_mse'),('skin_train_branch_v1','graph_always'))}


def metrics(e):
    return {'mean':float(np.mean(e)),'median':float(np.median(e)),'p95':float(np.quantile(e,.95)),
            'above5':float(np.mean(e>5)),'above10':float(np.mean(e>10))}


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    assert len(list((ROOT/'docs/benchmarks/skin_train_branch_v1').glob('*/result.json')))==36
    lock=OUT/'protocol_lock.json'
    bindings={'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(Path(__file__)),'pairs':PAIRS}
    # JSON converts tuples to lists; round-trip fixes the receipt representation.
    bindings=json.loads(json.dumps(bindings))
    if lock.exists():assert json.loads(lock.read_bytes())==bindings
    else:lock.write_text(json.dumps(bindings,indent=2)+'\n',encoding='utf8')
    records=[];gap=0.;cases=0;coveragecases=0;arrays={}
    for protocol in ['mixed','from_SLR','from_ipod']:
        for seed in [17,29,43]:
            for name,pair in PAIRS.items():
                values=[];digests=[];counts=[]
                for benchmark,arm in pair:
                    folder=ROOT/'experiments/runs'/benchmark/f'{protocol}__{arm}__s{seed}'
                    record=json.loads((ROOT/'docs/benchmarks'/benchmark/folder.name/'result.json').read_bytes())
                    assert sha(folder/'best.pt')==record['best_sha256']
                    with np.load(folder/'evaluation.npz') as a:
                        values.append({k:a[k].copy() for k in ['prediction','target','patient','site']})
                    digests.append(sha(folder/'evaluation.npz'))
                    counts.append(record.get('active_parameters',record['stored_parameters']))
                a,b=values
                for key in ['target','patient','site']:np.testing.assert_array_equal(a[key],b[key])
                p=.5*(a['prediction'].astype(float)+b['prediction'].astype(float))
                risk=delta_e00(a['prediction'],b['prediction']);e=delta_e00(p,a['target'])
                independent=np.array([scalar_de(x,y) for x,y in zip(p,a['target'],strict=True)])
                independent_risk=np.array([scalar_de(x,y) for x,y in zip(a['prediction'],b['prediction'],strict=True)])
                cases+=2*len(e);gap=max(gap,float(abs(e-independent).max()),float(abs(risk-independent_risk).max()))
                order=np.argsort(risk,kind='stable');coverage=[]
                for c in [1.,.95,.9,.8,.7,.6]:
                    n=int(np.ceil(c*len(e)));ix=order[:n]
                    coverage.append({'coverage':c,'accepted':n,**metrics(e[ix])});coveragecases+=1
                records.append({'protocol':protocol,'seed':seed,'method':name,'full':metrics(e),
                    'coverage':coverage,'constituent_means':[float(delta_e00(v['prediction'],a['target']).mean()) for v in values],
                    'member_prediction_sha256':digests,'combined_active_parameters':sum(counts)})
                arrays[f'{protocol}__{name}__s{seed}']=np.column_stack([p,e,risk])
    assert len(records)==45 and gap<1e-10
    local=ROOT/'data/processed/skin_branch_combination_v1';local.mkdir(parents=True,exist_ok=True);file=local/'predictions.npz'
    if file.exists():
        with np.load(file) as old:
            assert set(old.files)==set(arrays)
            for key,p in arrays.items():np.testing.assert_array_equal(old[key],p)
    else:np.savez_compressed(file,**arrays)
    result={'scope':'Adaptive SOURCE exploration; fixed equal averaging, uncalibrated disagreement; not independent test',
            'protocol_lock_sha256':sha(lock),'local_arrays_sha256':sha(file),'rows':records,
            'scalar_color_and_disagreement_cases':cases,'fixed_coverage_rows':coveragecases,'maximum_scalar_gap':gap}
    target=OUT/'summary.json'
    if target.exists():assert json.loads(target.read_bytes())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    lines=['# Equal-weight source combinations','',
        'Post-discovery source exploration. Fixed same-seed two-model nativeLab averaging;',
        'no tuned weights. Three seed scores averaged below, not a six-model ensemble.',
        'All members use their own same-camera validation checkpoint. No independent test.',
        'Two networks cost about twice one network; no ensemble novelty claim.','',
        '| Protocol | Pair | MeanDeltaE00 | At80% | p95 | Seed means |','|---|---|---:|---:|---:|---|']
    for protocol in ['mixed','from_SLR','from_ipod']:
        for name in PAIRS:
            rows=[r for r in records if r['protocol']==protocol and r['method']==name]
            mean=np.mean([r['full']['mean'] for r in rows]);risk80=np.mean([r['coverage'][3]['mean'] for r in rows]);p95=np.mean([r['full']['p95'] for r in rows])
            seedtext=', '.join(f"{r['full']['mean']:.4f}" for r in rows)
            lines.append(f'|{protocol}|{name}|{mean:.4f}|{risk80:.4f}|{p95:.4f}|{seedtext}|')
            print(json.dumps({'protocol':protocol,'pair':name,'mean':float(mean),'risk80':float(risk80),'seed_means':[r['full']['mean'] for r in rows]}))
    lines+=['','80%ranking uses uncalibrated inter-model disagreement; it may worsen error.',
        'This is not a learned expected-error head or coverage guarantee. All45combinations,',
        '270fixed-coverage rows and independent scalar checks are retained in [results](summary.json).',
        '[Protocol](../../research/skin_branch_combination_protocol_v1.md),',
        '[decision and limits](../../research/skin_spatial_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
