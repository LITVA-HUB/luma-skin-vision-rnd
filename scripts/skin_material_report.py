"""Report all image material controls with source-selection limits."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_material_train import OUT,RUN
from skin_material_model import ARMS
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==45
    paths=sorted(OUT.glob('*/result.json'));assert len(paths)==45
    records=[json.loads(p.read_bytes()) for p in paths]
    groups={};comparisons=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        for arm in ARMS:
            rr=sorted([r for r in records if r['protocol']==protocol and r['arm']==arm],key=lambda r:r['seed'])
            assert len(rr)==3
            groups[(protocol,arm)]={'mean':float(np.mean([r['full']['mean'] for r in rr])),
                'p95_mean':float(np.mean([r['full']['p95'] for r in rr])),
                'at80':float(np.mean([r['uncalibrated_disagreement_coverage'][3]['mean'] for r in rr])),
                'seed_means':[r['full']['mean'] for r in rr],
                'active_parameters':rr[0]['active_parameters'],'peak_mib':max(r['peak_train_allocated_mib'] for r in rr),
                'max_model_bytes':max(r['model_bytes'] for r in rr)}
        for arm,control in [('material','tangent'),('material_residual','tangent_residual'),('material_residual','direct')]:
            differences=[];patient=None
            for seed in (17,29,43):
                a=np.load(RUN/f'{protocol}__{arm}__s{seed}'/'evaluation.npz')
                b=np.load(RUN/f'{protocol}__{control}__s{seed}'/'evaluation.npz')
                np.testing.assert_array_equal(a['target'],b['target']);np.testing.assert_array_equal(a['patient'],b['patient'])
                differences.append(a['error']-b['error']);patient=a['patient']
            d=np.mean(differences,axis=0);pd=np.array([d[patient==p].mean() for p in sorted(set(patient))])
            rng=np.random.default_rng(52719)
            boot=pd[rng.integers(0,len(pd),size=(20000,len(pd)))].mean(1)
            comparisons.append({'protocol':protocol,'arm':arm,'control':control,
                'mean_image_error_difference':float(d.mean()),'mean_patient_difference':float(pd.mean()),
                'descriptive_patient_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                'all_three_seed_means_improve':all(x.mean()<0 for x in differences),
                'posthoc_repeated_source_not_confirmatory':True})
    historical=[]
    for protocol in ('mixed','from_SLR','from_ipod'):
        for method in ('plain_mse','mixture_mse'):
            ps=[ROOT/f'docs/benchmarks/skin_capture_v1/{protocol}__{method}__s{s}/result.json' for s in (17,29,43)]
            historical.append({'protocol':protocol,'method':method,'mean':float(np.mean([json.loads(p.read_bytes())['full']['mean'] for p in ps]))})
        ps=[ROOT/f'docs/benchmarks/skin_train_branch_v1/{protocol}__graph_always__s{s}/result.json' for s in (17,29,43)]
        historical.append({'protocol':protocol,'method':'training_only_graph','mean':float(np.mean([json.loads(p.read_bytes())['full']['mean'] for p in ps]))})
    curve_rows=[];fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    for ax,protocol in zip(axes,('mixed','from_SLR','from_ipod')):
        visible=[]
        for arm in ARMS:
            curves=[np.load(RUN/f'{protocol}__{arm}__s{s}'/'risk_curve.npz') for s in (17,29,43)]
            x=curves[0]['coverage'];y=np.mean([c['mean_error'] for c in curves],axis=0)
            for c in curves:np.testing.assert_array_equal(x,c['coverage'])
            ax.plot(x*100,y,label=arm)
            visible.extend(y[x>=.5])
            curve_rows.extend({'protocol':protocol,'arm':arm,'coverage':float(a),'mean_delta_e00':float(b)} for a,b in zip(x,y))
        ax.set(title=protocol,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00',xlim=(50,100))
        lo,hi=min(visible),max(visible);pad=max(.04,(hi-lo)*.15);ax.set_ylim(lo-pad,hi+pad)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7);fig.suptitle('Exploratory real-image source cohorts; uncalibrated hypothesis disagreement')
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=150);plt.close(fig)
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['protocol','arm','coverage','mean_delta_e00']);writer.writeheader();writer.writerows(curve_rows)
    summary={'scope':'45fixed source image fits; averages of separate seed scores, not ensembles',
        'groups':[dict(protocol=p,arm=a,**v) for (p,a),v in groups.items()],
        'descriptive_matched_comparisons':comparisons,'historical_locally_reproduced':historical,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths+[OUT/'audit.json',OUT/'prior_receipt.json',Path(__file__)]}}
    write(OUT/'summary.json',summary)
    lines=['# Real-image measured-material decoder screen', '',
        'All predictions use one image. Ground truth is actual MSKCC native instrument',
        'Lab (D65/10degree); DeltaE00 below is skin color error, not illuminant angle.',
        'These are heavily explored SOURCE development cohorts, not new independent',
        'tests. MSKCC TEST/CAL, ISSA reserved origins and UMINHO held-outs were not read.', '',
        'Five arms x three seeds x three protocols. All share the same parameter',
        'shapes, initialization, image data/resolution and80epoch fitting budget.',
        'Material/tangent have the same ISSA TRAIN-only prior. Tangent is its fixed',
        'first-order expansion, so the residual pair tests the nonlinear decoder',
        'against matched linear color geometry. Capture mode is never an input.', '',
        'Each table entry averages three separate seed scores, not ensemble predictions.',
        'The p95 column is the mean of three individual p95 values.', '',
        '| Protocol | Arm | Mean DeltaE00 | Mean seed p95 | Mean at80% | Seed means |',
        '|---|---|---:|---:|---:|---|']
    for (p,a),r in groups.items():lines.append(f"| {p} | {a} | {r['mean']:.4f} | {r['p95_mean']:.4f} | {r['at80']:.4f} | {', '.join(f'{v:.4f}' for v in r['seed_means'])} |")
    lines += ['', '## Strong historical controls (reproduced locally earlier)', '',
        '| Protocol | Method | Mean DeltaE00 |','|---|---|---:|']
    for r in historical:lines.append(f"| {r['protocol']} | {r['method']} | {r['mean']:.4f} |")
    lines += ['', '## Matched nonlinear mechanism comparisons', '',
        'Negative difference favors material. Intervals are descriptive patient-cluster',
        'bootstraps on repeatedly inspected cohorts, without multiplicity correction.',
        'They do not establish a confirmatory improvement.', '',
        '| Protocol | Arm vs control | Image-mean difference | Patient-mean95% interval | All3seeds improve |',
        '|---|---|---:|---|---|']
    for r in comparisons:
        lo,hi=r['descriptive_patient_bootstrap95'];lines.append(f"| {r['protocol']} | {r['arm']} vs {r['control']} | {r['mean_image_error_difference']:.4f} | [{lo:.4f},{hi:.4f}] | {r['all_three_seed_means_improve']} |")
    prior=json.loads((OUT/'prior_receipt.json').read_bytes())
    lines += ['', '## Color interface and scope', '',
        'The prior uses measured ISSA reflectance, not its2degree Lab labels. The',
        'decoder uses original CIE10degree CMFs and D65, with explicit interpolation',
        'and constant endpoint extrapolation of missing skin tails. This is a model',
        'assumption, not new spectral ground truth. On already-read TRAIN spectra',
        f"with extra measured bands, the derived tail sensitivity is mean{prior['derived_tail_sensitivity']['mean_delta_e00']:.5f},",
        f"p95{prior['derived_tail_sensitivity']['p95']:.5f}DeltaE00; neither branch is an instrument10degree validation.",
        'Residual variants can depart from the material model. Strong real-photo',
        'controls determine utility; low oracle spectral error does not.', '',
        'Original ISSA is CC BY4.0. Original CIE tables are CC BY-SA4.0; adapted',
        'integration constants retain attribution/share-alike. Do not describe a',
        'checkpoint bundling those buffers as unrestricted proprietary output.',
        '[CIE provenance and metadata discrepancy](../../data/provenance/cie_material_v1/README.md).', '',
        '## Uncalibrated risk and compute', '',
        'Ranking uses four-hypothesis dispersion, not calibrated expected error.',
        'A matched C+ error head and post-hoc calibration have not been demonstrated',
        'by this screen. Fixed coverage is not a deployed threshold guarantee.', '',
        '![Skin error and accepted coverage](risk_coverage.png)', '',
        'The CSV contains complete curves, including below50% shown outside the plot.',
        'All model states store937,521parameters. Direct uses929,297active; hard',
        'tangent/material934,437; residual arms937,521. No foundation model is needed.',
        f"Largest measured fit-process allocation: {max(r['peak_train_allocated_mib'] for r in records):.2f}MiB.",
        f"Largest model file: {max(r['model_bytes'] for r in records):,}bytes.",
        'Fit wall times include ordinary process overhead and some concurrent tests;',
        'they are not batch1latency. No new inference VRAM, ONNX or latency claim.', '',
        '## Verification', '',
        f"{audit['exact_color_prediction_arrays']}exact color arrays,45gate/hypothesis/free-coordinate sets,",
        f"{audit['independent_scalar_delta_e00_cases']}independent scalar color checks,",
        f"{audit['coverage_rows']}coverage rows and{audit['full_curve_points']}curve points verified.",
        'Frozen prior buffers, fitting-only target scales and same-camera epoch',
        'selection are checked. All matched initial state hashes agree.', '',
        'Independent MSKCC result remains unchanged: primary full4.4570/80%4.1591;',
        'ordinary fusion4.3005/4.1447 is stronger. Ordinary phone facial skin',
        'accuracy and a novel selective-system advantage remain unvalidated.', '']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps(summary['groups']))


if __name__=='__main__':main()
