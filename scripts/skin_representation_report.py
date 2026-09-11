"""Complete source reports for nuisance controls or rank-dependence context."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',choices=['nuisance','copula']);args=parser.parse_args()
    experiment='skin_'+args.experiment+'_v1';out=ROOT/'docs/benchmarks'/experiment;run=ROOT/'experiments/runs'/experiment
    audit=json.loads((out/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==36
    paths=sorted(out.glob('*/result.json'));assert len(paths)==36
    records=[json.loads(p.read_bytes()) for p in paths];bindings={str(p.relative_to(ROOT)):sha(p) for p in paths};groups=[]
    arms=['learned','fixed_grid','global','bias'] if args.experiment=='nuisance' else ['none','rgb_hist','copula','rank_only']
    def summary(rows,protocol,method):
        rows=sorted(rows,key=lambda r:r['seed']);assert [r['seed'] for r in rows]==[17,29,43]
        return {'protocol':protocol,'method':method,'seed_means':[r['full']['mean'] for r in rows],
            'mean':float(np.mean([r['full']['mean'] for r in rows])),'median':float(np.mean([r['full']['median'] for r in rows])),
            'p95':float(np.mean([r['full']['p95'] for r in rows]))}
    for protocol in ['mixed','from_SLR','from_ipod']:
        for arm in arms:
            rows=[r for r in records if r['protocol']==protocol and r['arm']==arm];g=summary(rows,protocol,arm)
            g.update({'risk80':float(np.mean([r['uncalibrated_dispersion_coverage'][3]['mean'] for r in rows])),
                      'nominal_inference_parameters':rows[0]['active_parameters'],
                      'nominal_training_parameters':rows[0].get('training_active_parameters',rows[0]['active_parameters']),
                      'max_fit_selection_mib':max(r['peak_train_allocated_mib'] for r in rows)})
            groups.append(g)
        for benchmark,arm,label in [('skin_spatial_v1','plain','historical_plain'),('skin_capture_v1','plain_mse','capture_plain'),
                                     ('skin_capture_v1','mixture_mse','capture_mixture'),('skin_train_branch_v1','graph_always','training_only_graph')]:
            base=ROOT/'docs/benchmarks'/benchmark;lock=json.loads((base/'source_lock.json').read_bytes())
            for name,digest in lock['bindings'].items():assert sha(ROOT/name)==digest
            rows=[]
            for seed in [17,29,43]:
                path=base/f'{protocol}__{arm}__s{seed}'/'result.json';rows.append(json.loads(path.read_bytes()));bindings[str(path.relative_to(ROOT))]=sha(path)
            groups.append(summary(rows,protocol,label))
    curves=out/'uncalibrated_risk_coverage.csv'
    with curves.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['protocol','method','seed','accepted','population','coverage','mean_delta_e00'])
        for r in records:
            with np.load(run/f"{r['protocol']}__{r['arm']}__s{r['seed']}"/'evaluation.npz') as a:
                e=a['error'][np.argsort(a['risk'],kind='stable')];risk=np.cumsum(e)/np.arange(1,len(e)+1)
                for k,v in enumerate(risk,1):writer.writerow([r['protocol'],r['arm'],r['seed'],k,len(e),k/len(e),float(v)])
    result={'scope':'Adaptive source research; native instrument skin color; seed score averaging is not ensembling',
            'groups':groups,'result_bindings':bindings,'audit_sha256':sha(out/'audit.json'),'source_lock_sha256':sha(out/'source_lock.json'),
            'curve_sha256':sha(curves),'report_script_sha256':sha(Path(__file__))}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    lines=[f'# {args.experiment}: real skin source mechanism screen','',
           'All36predeclared fits retained. Values average three separate seed scores, not an',
           'ensemble. Same source data/80epoch budget; native instrumentLab scored withCIEDE2000.',
           'The source roles and camera families have already been examined during research:',
           'this is not an independent test or ordinary facial-phone validation.','',
           '| Protocol | Method | MeanDeltaE00 | p95 | Seed means |','|---|---|---:|---:|---|']
    for g in groups:
        seeds=', '.join(f'{v:.4f}' for v in g['seed_means'])
        lines.append(f"|{g['protocol']}|{g['method']}|{g['mean']:.4f}|{g['p95']:.4f}|{seeds}|")
    lines+=['','Historical rows are LOCALLY REPRODUCED compatible source controls; bindings were',
            'rechecked. Strong previous results are retained, rather than comparing only to weak',
            'members of the current family.','',
            '## Risk and compute','',
            '| Protocol | Method | Mean at80% | Nominal inference params | Nominal training params | Max fit+selection MiB |',
            '|---|---|---:|---:|---:|---:|']
    for g in groups:
        if g['method'] in arms:
            lines.append(f"|{g['protocol']}|{g['method']}|{g['risk80']:.4f}|{g['nominal_inference_parameters']}|{g['nominal_training_parameters']}|{g['max_fit_selection_mib']:.2f}|")
    lines+=['','Risk is uncalibrated patch-vote dispersion, not expected error or an accept guarantee.',
            'All100/95/90/80/70/60%metrics are in run results; [full curves](uncalibrated_risk_coverage.csv).']
    if args.experiment=='nuisance':
        lines+=['','All inference uses the same plain core. Fixed/global operators are image-independent',
            'but their latent residuals are image-dependent. Bias has only256functional offset',
            'dimensions despite its nominal65,536matrix parameters.9learned-control final weight',
            'sets exactly match the previous training-only graph experiment.','',
            'A declared [test-only precision amendment](test_precision_amendment/amendment.json)',
            'checks mathematical constant preservation inFP64 rather than an overly tightFP32',
            'absolute tolerance. Original test/lock archived. Model, data and fitting unchanged.']
    else:
        lines+=['','Copula and RGB-histogram context have exactly matched architecture and capacity.',
            'None has a constant branch input, so nominal parameter equality does not imply equal',
            'informative capacity. Rank-only intentionally deletes absolute-color information.',
            '1,230independent count-CDF histogram and1,536patch-profile checks are exact.',
            'The16strict-monotone input probes have zero histogram change; channel mixing and',
            'clipping do change it. This is not arbitrary-camera invariance or an accuracy test.']
    lines+=['','## Verification','',
            f"{audit['exact_prediction_arrays']}color and{audit['exact_risk_arrays']}risk arrays exactly replayed;",
            f"{audit['independent_scalar_delta_e00_cases']}independent scalar cases/{audit['independent_coverage_rows']}coverage rows checked.",
            'Fitting-only target scales, identical per-seed initializations and same-camera epoch',
            'selection verified. TEST/CAL and UMINHO held-out endpoints were not used.','',
            f'[Protocol](../../research/skin_{args.experiment}_protocol_v1.md), [audit](audit.json),',
            '[decision](../../research/skin_representation_next_decision.md).']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for g in groups:
        if g['method'] in arms:print(json.dumps(g))


if __name__=='__main__':main()
