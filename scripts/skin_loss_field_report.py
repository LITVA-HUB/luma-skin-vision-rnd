"""All skin-loss fields, negative ablations and target-free grid controls."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from luma_skin_vision.color import delta_e00
from skin_loss_field_train import OUT,RUN,PRIOR
from skin_loss_field import ARMS
from skin_mskcc_data import ROOT,sha
from skin_mskcc_pixels import load
from skin_pair_train import subset,rows,write
from skin_mskcc_summary_pilot import summarize
from skin_mskcc_audit import scalar_de


def nearest(points,grid):
    return np.concatenate([delta_e00(p[:,None],grid[None]).argmin(1) for p in np.array_split(points,int(np.ceil(len(points)/16)))])


def main():
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS' and audit['fits']==36
    paths=sorted(OUT.glob('*/result.json'));records=[json.loads(p.read_bytes()) for p in paths];assert len(records)==36
    val=load('validation');groups={};curves=[];comparisons=[];grid_controls={};gap=0.;scalar=0
    for protocol in ('mixed','from_SLR','from_ipod'):
        ev=val if protocol=='mixed' else subset(val,val['device']!=protocol.removeprefix('from_'))
        grid=np.load(PRIOR/f'{protocol}.npz')['grid'].astype(np.float32)
        oracle_idx=nearest(ev['target'],grid);oracle=grid[oracle_idx]
        e=np.array([scalar_de(a,b) for a,b in zip(oracle,ev['target'])]);scalar+=len(e)
        control={'known_target_nearest_grid_oracle':summarize(e,rows(ev)),'direct_projected':[]}
        arrays={}
        for arm in ARMS:
            rr=sorted([r for r in records if r['protocol']==protocol and r['arm']==arm],key=lambda r:r['seed'])
            aa=[dict(np.load(RUN/f'{protocol}__{arm}__s{s}'/'evaluation.npz')) for s in (17,29,43)];arrays[arm]=aa
            for endpoint in ('primary','secondary'):
                groups[f'{protocol}/{arm}/{endpoint}']={
                    'mean':float(np.mean([r['scores'][endpoint]['full']['mean'] for r in rr])),
                    'median':float(np.mean([r['scores'][endpoint]['full']['median'] for r in rr])),
                    'p95':float(np.mean([r['scores'][endpoint]['full']['p95'] for r in rr])),
                    'at80':float(np.mean([r['scores'][endpoint]['coverage'][3]['mean'] for r in rr])),
                    'seed_means':[r['scores'][endpoint]['full']['mean'] for r in rr],
                    'negative_risk_fraction':float(np.mean([r['negative_risk_fraction'] for r in rr])),
                    'negative_weight_fraction':float(np.mean([r['negative_weight_fraction'] for r in rr]))}
                curve=np.mean([a[endpoint+'_curve'] for a in aa],axis=0)
                for i,risk in enumerate(curve):curves.append({'protocol':protocol,'method':arm+'/'+endpoint,'coverage':(i+1)/len(curve),'mean_delta_e00':float(risk)})
        for seed,a in zip((17,29,43),arrays['direct']):
            idx=nearest(a['prediction'],grid);prediction=grid[idx]
            error=delta_e00(prediction,ev['target']);ind=np.array([scalar_de(x,y) for x,y in zip(prediction,ev['target'])]);scalar+=len(ind)
            gap=max(gap,float(np.max(abs(ind-error))))
            control['direct_projected'].append({'seed':seed,**summarize(ind,rows(ev))})
            np.savez(RUN/f'{protocol}__direct__s{seed}'/'grid_projection.npz',prediction=prediction,error=error,indices=idx)
        grid_controls[protocol]=control
        for arm,base in [('risk_simplex','soft_ce'),('risk_affine','soft_ce'),('risk_affine','direct')]:
            aa,bb=arrays[arm],arrays[base]
            diff=np.mean([a['primary_error']-b['primary_error'] for a,b in zip(aa,bb)],axis=0)
            people=np.unique(ev['patient']);cluster=np.array([diff[ev['patient']==p].mean() for p in people])
            rng=np.random.default_rng(51031);boot=cluster[rng.integers(0,len(cluster),(10000,len(cluster)))].mean(1)
            comparisons.append({'protocol':protocol,'candidate':arm,'control':base,'image_mean_difference':float(diff.mean()),
                'descriptive_patient_interval':np.quantile(boot,[.025,.975]).tolist(),
                'all3seeds_improve':all(a['primary_error'].mean()<b['primary_error'].mean() for a,b in zip(aa,bb))})
    assert gap<1e-9
    summary={'scope':'Repeated SOURCE development; no independent test', 'groups':groups,'comparisons':comparisons,
        'posthoc_grid_controls':grid_controls,'additional_scalar_cases':scalar,'maximum_scalar_gap':gap,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'audit.json',OUT/'source_lock.json',OUT/'palette_audit.json']}}
    write(OUT/'summary.json',summary)
    with (OUT/'risk_coverage.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,protocol in zip(axes,('mixed','from_SLR','from_ipod')):
        for arm in ARMS:
            cc=[c for c in curves if c['protocol']==protocol and c['method']==arm+'/primary' and c['coverage']>=.5]
            ax.plot([100*c['coverage'] for c in cc],[c['mean_delta_e00'] for c in cc],label=arm)
        ax.set(title=protocol,xlabel='Accepted coverage (%)',ylabel='Mean skin DeltaE00');ax.grid(alpha=.25)
    axes[-1].legend();fig.suptitle('Real skin source loss fields; uncalibrated scores; affine score may be negative')
    fig.tight_layout();fig.savefig(OUT/'risk_coverage.png',dpi=145);plt.close(fig)
    lines=['# Native skin-color loss-field experiment','',
        '36locally reproduced real-image fits; native MSKCC instrument Lab ground truth.',
        'All color errors are DeltaE00. No fresh TEST/CAL or ordinary-phone result.',
        'Source validation has been inspected repeatedly. Prefit checkpoint:',
        '`checkpoint/skin-loss-field-pretrain-2026-09-11`.','',
        'Each result averages three separate seed scores, not an ensemble. For atom',
        'models the primary answer minimizes a learned field on15,625fixed candidate',
        'colors; secondary is the atom-weighted Lab at the same selected checkpoint.',
        'Direct is the ordinary continuous regressor. Both field losses share a full',
        'Gram factor; no spectral truncation or invented instrument reference is used.','',
        '| Protocol | Arm | Primary mean | Median | p95 | At80% | Secondary mean | Negative risk |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for protocol in ('mixed','from_SLR','from_ipod'):
        for arm in ARMS:
            g=groups[f'{protocol}/{arm}/primary'];s=groups[f'{protocol}/{arm}/secondary']
            lines.append(f"| {protocol} | {arm} | {g['mean']:.4f} | {g['median']:.4f} | {g['p95']:.4f} | {g['at80']:.4f} | {s['mean']:.4f} | {100*g['negative_risk_fraction']:.2f}% |")
    lines+=['','soft_ce is a standard smoothed categorical objective (fixed width2DeltaE00).',
        'risk_simplex learns candidate loss with nonnegative unit-sum weights.',
        'risk_affine removes nonnegativity; it is not a probability estimator.',
        'Negative predicted risks, where present, are unsupported expected errors.',
        'No score is post-hoc calibrated or a per-image error bound.','',
        '## Matched primary comparisons','',
        'Patient intervals are descriptive on reused source cohorts, without',
        'multiplicity adjustment. Camera protocols also change people and capture mode.','',
        '| Protocol | Candidate vs control | Mean difference | Patient95% interval | All3seeds |',
        '|---|---|---:|---|---|']
    for c in comparisons:
        lo,hi=c['descriptive_patient_interval']
        lines.append(f"| {c['protocol']} | {c['candidate']} vs {c['control']} | {c['image_mean_difference']:.4f} | [{lo:.4f},{hi:.4f}] | {c['all3seeds_improve']} |")
    lines+=['','## Post-hoc grid limitation checks','',
        'Projecting the frozen ordinary answer onto the same grid uses no true color',
        'during projection. The separately labeled oracle uses the true reference and',
        'is only a grid approximation diagnostic, never an image-accuracy result.','',
        '| Protocol | Direct projected mean | Known-target oracle mean | Oracle p95 |',
        '|---|---:|---:|---:|']
    for protocol,c in grid_controls.items():
        o=c['known_target_nearest_grid_oracle'];mean=np.mean([r['mean'] for r in c['direct_projected']])
        lines.append(f"| {protocol} | {mean:.4f} | {o['mean']:.4f} | {o['p95']:.4f} |")
    lines+=['','![Risk and coverage](risk_coverage.png)','',
        'All six fixed coverages and full curves are retained. Expected losses over a',
        'finite TRAIN color dictionary need not describe a new-camera distribution.',
        'Known-target grid feasibility does not imply inverse image identifiability.','',
        '## Compute, verification and remaining scope','',
        f"Stored parameters range{min(r['stored_parameters'] for r in records):,}-{max(r['stored_parameters'] for r in records):,}.",
        f"Maximum learned checkpoint{max(r['checkpoint_bytes'] for r in records):,}bytes; shared fixed palette up to{max(r['fixed_palette_bytes'] for r in records):,}bytes.",
        f"Maximum fit-process GPU allocation{max(r['fit_peak_allocated_mib'] for r in records):.2f}MiB.",
        'Fixed tables are not trainable parameters; they are required by the field',
        'decoder. Direct does not mathematically need the table despite shared harness.',
        'No latency, ONNX, or deployment promotion. No foundation inference/camera input.','',
        f"{audit['exact_color_replay_arrays']}exact color arrays; {audit['independent_scalar_color_cases']}scalar color cases;",
        f"{audit['coverage_rows']}coverage rows and{audit['curve_points']}curve points; scalar palette audit and full Gram checks pass.",
        'FP64 field products check the selected action with explicit FP32 roundoff bounds.',
        'Original MSKCC CC-BY; no external data, weights or CIE tables adopted here.',
        'Independent benchmark remains unchanged4.4570/80%4.1591; ordinary fusion',
        '4.3005/4.1447 is stronger. Novel superiority and facial-phone accuracy unproven.','']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(groups))


if __name__=='__main__':main()
