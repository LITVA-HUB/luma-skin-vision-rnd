"""Report all frozen density endpoints without selecting a favorable protocol."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_distribution_train import OUT,RUN
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==36
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==36
    records=[json.loads(p.read_bytes()) for p in paths];groups={};curves=[];comparisons=[]
    protocols=['mixed','from_SLR','from_ipod']
    versions=[('mse_mode','mean'),('mse','mean'),('gaussian','mean'),('gaussian','decision3'),
        ('mdn4','mean'),('mdn4','decision3'),('mdn4','decision2')]
    def runs(protocol,arm):return sorted([r for r in records if r['protocol']==protocol and r['arm']==arm],key=lambda r:r['seed'])
    def arrays(protocol,arm):
        return [dict(np.load(RUN/f'{protocol}__{arm}__s{s}'/'evaluation.npz')) for s in (17,29,43)]
    for protocol in protocols:
        for arm,endpoint in versions:
            rr=runs(protocol,arm);ss=[r['scores'][endpoint] for r in rr]
            key=f'{protocol}/{arm}/{endpoint}'
            aa=arrays(protocol,arm)
            groups[key]={'mean':float(np.mean([v['full']['mean'] for v in ss])),
                'median':float(np.mean([v['full']['median'] for v in ss])),
                'p95':float(np.mean([v['full']['p95'] for v in ss])),
                'at80':float(np.mean([v['coverage'][3]['mean'] for v in ss])),
                'seed_means':[v['full']['mean'] for v in ss],
                'coverage':[{'coverage':ss[0]['coverage'][i]['requested_coverage'],
                    'mean':float(np.mean([v['coverage'][i]['mean'] for v in ss])),
                    'p95':float(np.mean([v['coverage'][i]['p95'] for v in ss])),
                    'above10':float(np.mean([v['coverage'][i]['above_10_fraction'] for v in ss]))} for i in range(6)]}
            curve=np.mean([a[endpoint+'_curve'] for a in aa],axis=0)
            for i,risk in enumerate(curve):curves.append({'protocol':protocol,'method':arm+'/'+endpoint,'coverage':(i+1)/len(curve),'mean_delta_e00':float(risk)})
        contrasts=[('mse','mean','mse_mode','mean'),('gaussian','mean','mse_mode','mean'),
            ('mdn4','mean','mse_mode','mean'),('mdn4','decision3','gaussian','decision3'),
            ('mdn4','decision3','mdn4','mean')]
        for a,ea,b,eb in contrasts:
            aa,bb=arrays(protocol,a),arrays(protocol,b)
            for x,y in zip(aa,bb):np.testing.assert_array_equal(x['patient'],y['patient'])
            delta=np.mean([x[ea+'_error']-y[eb+'_error'] for x,y in zip(aa,bb)],axis=0)
            people=np.unique(aa[0]['patient']);cluster=np.array([delta[aa[0]['patient']==p].mean() for p in people])
            rng=np.random.default_rng(73011);boot=cluster[rng.integers(0,len(cluster),(10000,len(cluster)))].mean(1)
            comparisons.append({'protocol':protocol,'candidate':a+'/'+ea,'control':b+'/'+eb,
                'image_mean_difference':float(delta.mean()),'patient_mean_difference':float(cluster.mean()),
                'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist(),
                'all_three_seed_means_improve':all(x[ea+'_error'].mean()<y[eb+'_error'].mean() for x,y in zip(aa,bb))})
    integration=json.loads((OUT/'integration_diagnostic.json').read_bytes()) if (OUT/'integration_diagnostic.json').exists() else None
    if integration:assert len(integration['records'])==18
    source_files=[Path(__file__),OUT/'audit.json',OUT/'source_lock.json']
    if integration:source_files.append(OUT/'integration_diagnostic.json')
    summary={'scope':'REPRODUCED SOURCE ONLY; repeated validation; no confirmatory test',
        'groups':groups,'comparisons':comparisons,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in source_files}}
    write(OUT/'summary.json',summary)
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,3,figsize=(16,4.6))
    for ax,protocol in zip(axes,protocols):
        for arm,endpoint in versions:
            if (arm,endpoint) in [('gaussian','decision3'),('mdn4','decision2')]:continue
            cc=[c for c in curves if c['protocol']==protocol and c['method']==arm+'/'+endpoint and c['coverage']>=.5]
            ax.plot([100*c['coverage'] for c in cc],[c['mean_delta_e00'] for c in cc],label=arm+'/'+endpoint)
        ax.set(title=protocol,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.25)
    axes[-1].legend(fontsize=8);fig.suptitle('Real skin source cohorts; uncalibrated distribution risk / baseline disagreement')
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=145);plt.close(fig)
    lines=['# Conditional skin-color distribution experiment','',
        '36 locally reproduced fits on real MSKCC skin images and actual native instrument Lab.',
        'All numbers are DeltaE00, not illumination degrees. These source validation cohorts',
        'have been inspected repeatedly; neither new-person independent confirmation nor',
        'ordinary unseen-phone facial accuracy is established. No TEST/CAL endpoints loaded.','',
        'The prefit protocol/code checkpoint is `checkpoint/skin-distribution-pretrain-2026-09-11`.',
        'Four training objectives share one compact backbone and parameter shapes. Three seeds,',
        '80epochs, same data/resolution, target scales and same-camera point-error selection.',
        'Mixture density networks and conditional risk decisions are existing ideas.','',
        '## Actual skin-color errors','',
        'Each cell averages three separate seed scores; this is not an ensemble. The p95',
        'column averages individual seed p95 values. Lower is better.','',
        '| Protocol | Training / decision | Mean | Median | p95 | At80% |',
        '|---|---|---:|---:|---:|---:|']
    for key,v in groups.items():
        protocol,arm,endpoint=key.split('/')
        lines.append(f"| {protocol} | {arm}/{endpoint} | {v['mean']:.4f} | {v['median']:.4f} | {v['p95']:.4f} | {v['at80']:.4f} |")
    lines+=['','mse_mode is the strong ordinary control; mse removes auxiliary capture-mode',
        'supervision. gaussian is a single diagonal density; mdn4 has four components.',
        'decision3 minimizes approximate expected DeltaE00 among a frozen finite candidate',
        'set using order3 Gaussian quadrature. decision2 is fixed numerical sensitivity,',
        'not a validation-selected alternative. This is not the exact continuous Bayes optimum.','',
        '## Matched contrasts','',
        'Intervals resample patient means and are descriptive on heavily reused source data.',
        'They are not simultaneous confidence intervals or confirmatory discovery tests.','',
        '| Protocol | Candidate vs control | Image mean difference | Patient95% interval | All3seeds improve |',
        '|---|---|---:|---|---|']
    for c in comparisons:
        ci=c['descriptive_patient_interval']
        lines.append(f"| {c['protocol']} | {c['candidate']} vs {c['control']} | {c['image_mean_difference']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {c['all_three_seed_means_improve']} |")
    lines+=['','## Risk and limitations','',
        '![Risk and coverage](risk_coverage.png)','',
        'Full curves are in risk_coverage.csv. All fixed coverage values including p95 and',
        'catastrophic errors are in summary.json and each result.json. Density expected',
        'error is uncalibrated; baseline ranking uses hypothesis dispersion, not expected',
        'error. These are not matched calibrated C+ results or per-image guarantees.',
        'Distribution misspecification, tiny source cohorts and Gaussian support outside',
        'physically possible skin colors limit interpretation. Camera transfers also change',
        'people/capture distributions. No single cause of transfer error is identified.','',
        'Strong historical source controls: mixed direct3.4406; SLR-to-iPod material4.9083;',
        'reverse training-only graph4.9736. These are locally measured, not author-reported',
        'numbers or independent benchmark comparisons. All remain part of the comparison.','',
        '## Compute and verification','',
        f"Stored parameters: {records[0]['stored_parameters']:,}; maximum checkpoint: {max(r['checkpoint_bytes'] for r in records):,}bytes.",
        f"Maximum fit-process GPU allocation: {max(r['fit_peak_allocated_mib'] for r in records):.2f}MiB.",
        'MSE scale-head parameters are inactive. No foundation inference, camera-ID input,',
        'or test-time calibration. CPU quadrature work is additional inference cost;',
        'no new batch1latency, inference VRAM, export or end-to-end speed claim is made.','',
        f"Audit: {audit['exact_color_replay_arrays']}exact color arrays; {audit['independent_scalar_color_cases']}independent scalar color cases;",
        f"{audit['coverage_rows']}coverage rows; {audit['curve_points']}curve points; {audit['manual_quadrature_scalar_cases']}manual quadrature scalar cases.",
        'Fit-only scalers, shared initialization and same-camera checkpoint selection checked.',
        'Tests compare density likelihood to torch.distributions and independently check',
        'quadrature mean/covariance. This verifies calculations, not scientific superiority.','',
        'Original MSKCC CC-BY data; no newly imported third-party weights/code/constants.',
        'Independent test remains primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447.',
        'Product precision and a distinctive superior mechanism remain unproven.','']
    if integration:
        lines+=['## Post-hoc numerical integration diagnostic','',
            'Weights and the finite candidate family remain unchanged. Antithetic Sobol',
            '1024/4096nodes per component are fixed sensitivity checks, not replacements',
            'for the frozen primary results or guarantees of an exact integral.','',
            '| Protocol | Density | Mean1024 | Mean4096 | At80%4096 |',
            '|---|---|---:|---:|---:|']
        for protocol in protocols:
            for arm in ('gaussian','mdn4'):
                rr=[r for r in integration['records'] if r['protocol']==protocol and r['arm']==arm]
                a,b=[np.mean([r['scores'][n]['full']['mean'] for r in rr]) for n in ('1024','4096')]
                selective=np.mean([r['scores']['4096']['coverage'][3]['mean'] for r in rr])
                lines.append(f'| {protocol} | {arm} | {a:.4f} | {b:.4f} | {selective:.4f} |')
        lines+=['','More accurate integration does not rescue the four-component density here.',
            f"{integration['independent_scalar_cases']}additional independent scalar cases pass; antithetic means checked.",
            'Full rule differences and empirical normal covariance matrices are in',
            '[integration_diagnostic.json](integration_diagnostic.json).', '',
            '[Research decision and competing next experiments](../../research/skin_distribution_next_decision.md).','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(groups))


if __name__=='__main__':main()
