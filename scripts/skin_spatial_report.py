"""Summarize all predeclared source fits; seed means are not ensembles."""
import csv,json
from pathlib import Path
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_spatial_train import OUT,RUN
from skin_spatial_model import ARMS


def main():
    records=[json.loads(p.read_bytes()) for p in sorted(OUT.glob('*/result.json'))]
    assert len(records)==54
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==54
    groups=[]
    for protocol in ['mixed','from_SLR','from_ipod']:
        for arm in ARMS:
            rows=sorted([r for r in records if r['protocol']==protocol and r['arm']==arm],key=lambda r:r['seed'])
            assert [r['seed'] for r in rows]==[17,29,43]
            groups.append({'protocol':protocol,'arm':arm,'seed_means':[r['full']['mean'] for r in rows],
                'mean_over_seeds':float(np.mean([r['full']['mean'] for r in rows])),
                'median_over_seeds':float(np.mean([r['full']['median'] for r in rows])),
                'p95_over_seeds':float(np.mean([r['full']['p95'] for r in rows])),
                'mean_risk80_over_seeds':float(np.mean([r['uncalibrated_dispersion_coverage'][3]['mean'] for r in rows])),
                'active_parameters':rows[0]['active_parameters'],'stored_parameters':rows[0]['stored_parameters'],
                'maximum_train_mib':max(r['peak_train_allocated_mib'] for r in rows),
                'mean_fit_seconds':float(np.mean([r['elapsed_seconds'] for r in rows]))})
    curve=OUT/'uncalibrated_risk_coverage.csv'
    with curve.open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(['protocol','arm','seed','accepted','population','coverage','mean_delta_e00'])
        for r in records:
            with np.load(RUN/f"{r['protocol']}__{r['arm']}__s{r['seed']}"/'evaluation.npz') as a:
                order=np.argsort(a['risk'],kind='stable');errors=a['error'][order];means=np.cumsum(errors)/np.arange(1,len(errors)+1)
                for i,m in enumerate(means,1):writer.writerow([r['protocol'],r['arm'],r['seed'],i,len(errors),i/len(errors),float(m)])
    result={'scope':'Source exploration; same seed score averaging, no ensemble or independent test claim',
            'groups':groups,'source_lock_sha256':sha(OUT/'source_lock.json'),'audit_sha256':sha(OUT/'audit.json'),
            'curves_sha256':sha(curve),'report_script_sha256':sha(Path(__file__))}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    lines=['# Real-photo spatial and recurrent graph source experiment','',
           '54predeclared fits, all retained. Values are native instrument-referenceDeltaE00.',
           'Each entry averages three separately trained seed scores; this is not an ensemble.',
           'Source VAL was used for checkpoint selection. Camera transfer is source exploration,',
           'not a new independent unseen-camera or ordinary facial-phone benchmark.','',
           '| Protocol | Method | Mean | Mean at80% | p95 | Seed means |',
           '|---|---|---:|---:|---:|---|']
    for g in groups:
        seeds=', '.join(f'{v:.4f}' for v in g['seed_means'])
        lines.append(f"|{g['protocol']}|{g['arm']}|{g['mean_over_seeds']:.4f}|{g['mean_risk80_over_seeds']:.4f}|{g['p95_over_seeds']:.4f}|{seeds}|")
    lines+=['','80%ranking uses uncalibrated nativeLab patch-vote dispersion. It is not an',
            'expected-error head, a calibrated accept threshold, or a validated coverage guarantee.',
            'Full100/95/90/80/70/60%metrics are in every run result; all integer coverage points',
            'are in [the curve CSV](uncalibrated_risk_coverage.csv).','',
            '## Matched mechanism comparisons','']
    lookup={(g['protocol'],g['arm']):g for g in groups}
    for protocol in ['mixed','from_SLR','from_ipod']:
        for proposed,control in [('graph1','conv1'),('graph3','conv3'),('graph3','plain'),('graph3','graph3_scrambled')]:
            p=lookup[protocol,proposed];c=lookup[protocol,control]
            wins=int(np.sum(np.array(p['seed_means'])<c['seed_means']))
            lines.append(f"- {protocol}: {proposed} minus {control} = {p['mean_over_seeds']-c['mean_over_seeds']:+.4f}DeltaE00; lower in {wins}/3seeds.")
    lines+=['','## Compute accounting','',
            '| Method | Stored parameters | Active parameters | Max fit+selection VRAM MiB |',
            '|---|---:|---:|---:|']
    for arm in ARMS:
        gs=[g for g in groups if g['arm']==arm];g=gs[0]
        lines.append(f"|{arm}|{g['stored_parameters']}|{g['active_parameters']}|{max(v['maximum_train_mib'] for v in gs):.2f}|")
    lines+=['','Unused modules preserve exactly matched initialization/storage across arms; active',
            'parameters are the honest capacity comparison. Conventional spatial controls have',
            'about0.23%more active parameters than graph variants. Plain has fewer parameters.',
            'No pretrained weights, spectral pseudo-labels, camera identity or extra training data.',
            'Only two already verified source cache files are loaded.','',
            '## Verification','',
            f"{audit['exact_prediction_arrays']}exact color arrays and {audit['exact_risk_arrays']}risk arrays replayed;",
            f"{audit['independent_scalar_delta_e00_cases']}independent scalarDeltaE00 cases and {audit['independent_coverage_rows']}coverage rows checked.",
            f"Maximum metric discrepancy {audit['maximum_metric_gap']:.3g}.",
            'Source hashes, fitting-only target scales, patient/camera boundaries, identical',
            'initial states per seed, and same-camera epoch selection verified.','',
            '[Protocol](../../research/skin_spatial_protocol_v1.md), [audit](audit.json),',
            '[decision](../../research/skin_spatial_next_decision.md).']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for g in groups:print(json.dumps(g))


if __name__=='__main__':main()
