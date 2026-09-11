"""All follow-up fits plus strong compatible historical source controls."""
import csv,json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_train_branch_train import OUT,RUN
from skin_train_branch_model import ARMS


def summarize_rows(rows,protocol,method):
    rows=sorted(rows,key=lambda r:r['seed']);assert [r['seed'] for r in rows]==[17,29,43]
    return {'protocol':protocol,'method':method,'seed_means':[r['full']['mean'] for r in rows],
            'mean':float(np.mean([r['full']['mean'] for r in rows])),
            'median':float(np.mean([r['full']['median'] for r in rows])),
            'p95':float(np.mean([r['full']['p95'] for r in rows]))}


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==36
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==36
    records=[json.loads(p.read_bytes()) for p in paths];groups=[];bindings={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for protocol in ['mixed','from_SLR','from_ipod']:
        for arm in ARMS:
            rows=[r for r in records if r['protocol']==protocol and r['arm']==arm]
            g=summarize_rows(rows,protocol,arm)
            g['mean_risk80']=float(np.mean([r['uncalibrated_dispersion_coverage'][3]['mean'] for r in rows]))
            g['active_inference_parameters']=rows[0]['active_parameters'];g['training_active_parameters']=rows[0]['training_active_parameters']
            g['max_train_mib']=max(r['peak_train_allocated_mib'] for r in rows);groups.append(g)
        for benchmark,arm,method in [('skin_spatial_v1','plain','original_plain'),('skin_spatial_v1','conv3','ordinary_conv3'),
                                     ('skin_capture_v1','plain_mse','capture_plain_mse'),('skin_capture_v1','mixture_mse','capture_mixture_mse')]:
            source=ROOT/'docs/benchmarks'/benchmark
            lock=json.loads((source/'source_lock.json').read_bytes())
            for name,digest in lock['bindings'].items():assert sha(ROOT/name)==digest
            rows=[]
            for seed in [17,29,43]:
                p=source/f'{protocol}__{arm}__s{seed}'/'result.json';rows.append(json.loads(p.read_bytes()));bindings[str(p.relative_to(ROOT))]=sha(p)
            groups.append(summarize_rows(rows,protocol,method))
    curves=OUT/'uncalibrated_risk_coverage.csv'
    with curves.open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['protocol','method','seed','accepted','population','coverage','mean_delta_e00'])
        for r in records:
            with np.load(RUN/f"{r['protocol']}__{r['arm']}__s{r['seed']}"/'evaluation.npz') as a:
                e=a['error'][np.argsort(a['risk'],kind='stable')];cumulative=np.cumsum(e)/np.arange(1,len(e)+1)
                for k,m in enumerate(cumulative,1):w.writerow([r['protocol'],r['arm'],r['seed'],k,len(e),k/len(e),float(m)])
    result={'scope':'Post-discovery source follow-up; no independent test, no ensemble; strong local controls retained',
            'groups':groups,'result_bindings':bindings,'audit_sha256':sha(OUT/'audit.json'),
            'source_lock_sha256':sha(OUT/'source_lock.json'),'curve_sha256':sha(curves),'script_sha256':sha(Path(__file__))}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    lines=['# Training-only spatial branch: real skin source follow-up','',
        '36predeclared follow-up fits, with zero branch steps at EVERY selection/evaluation endpoint.',
        'The earlier post-hoc discovery was on these same source roles: this is not an independent',
        'test or an unbiased estimate after architecture search. Targets are actual instrument',
        'nativeLab. No camera ID, spectral pseudo-label, extra image or pretrained weight.','',
        'Each number averages three separately trained seed scores; not an ensemble.','',
        '| Protocol | Method | MeanDeltaE00 | p95 | Seed means |','|---|---|---:|---:|---|']
    for g in groups:
        seedtext=', '.join(f'{x:.4f}' for x in g['seed_means'])
        lines.append(f"|{g['protocol']}|{g['method']}|{g['mean']:.4f}|{g['p95']:.4f}|{seedtext}|")
    lines+=['','Historical controls above are LOCALLY REPRODUCED, with code/data bindings rechecked.',
            'Capture-plainMSE and capture-mixtureMSE use approximately0.929Mparameters and the',
            'same source data, resolution and80epoch budget; their stronger results are retained.',
            'Always versus drop differs only in branch usage during fitting (.5enabled per batch',
            'for drop). All four methods deploy the same924,932parameter plain inference core.',
            'Training auxiliaries remain in the checkpoint; stored count993,287is not active size.','',
            '## Uncalibrated risk diagnostic','',
            '| Protocol | Method | Mean at80% | Training active parameters | Max fit+selection MiB |',
            '|---|---|---:|---:|---:|']
    for g in groups:
        if g['method'] not in ARMS:continue
        lines.append(f"|{g['protocol']}|{g['method']}|{g['mean_risk80']:.4f}|{g['training_active_parameters']}|{g['max_train_mib']:.2f}|")
    lines+=['','Dispersion ranking is uncalibrated and may worsen accepted-image error. All fixed',
            'coverage endpoints are in run results, with full [risk curves](uncalibrated_risk_coverage.csv).',
            'No expected-error or guaranteed acceptance claim follows.','',
            '## Verification','',
            f"{audit['exact_prediction_arrays']}color and{audit['exact_risk_arrays']}risk arrays exactly replayed;",
            f"{audit['independent_scalar_delta_e00_cases']}independent scalar cases/{audit['independent_coverage_rows']}coverage rows pass.",
            f"{audit['identical_always_arm_training_trajectories']}always-arm final parameter sets exactly equal the prior spatial fits.",
            'This verifies that their training trajectory was unchanged; only inference and',
            'same-camera checkpoint selection differ. Fit-only target scales and subject/camera',
            'boundaries pass. No MSKCC TEST/CAL or UMINHO held-out data were loaded.','',
            '[Protocol](../../research/skin_train_branch_protocol_v1.md), [audit](audit.json),',
            '[decision](../../research/skin_spatial_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for g in groups:print(json.dumps(g))


if __name__=='__main__':main()
