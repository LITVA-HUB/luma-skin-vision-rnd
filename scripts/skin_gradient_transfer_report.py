"""Aggregate TRAIN mechanism evidence; never a replacement accuracy table."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_gradient_transfer_run import OUT
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    data=json.loads((OUT/'results.json').read_bytes());audit=json.loads((OUT/'audit.json').read_bytes())
    assert audit['status']=='PASS'
    for f,h in audit['bindings'].items():assert sha(ROOT/f)==h
    summaries=[];steps=[];lines=[
        '# TRAIN-only skin gradient/update diagnostic',
        '', 'No new neural fit or held-out accuracy improvement. Six frozen929,297-parameter models; '
        '24 partitions,48 objective cases,288 transient steps. Original TRAIN only. '
        'Original MSKCC CC-BY instrument skin references; no additional data/weights.',
        '', '## Does person identity explain gradient conflict?',
        '', 'Numbers below describe standardized native-Lab MSE gradients, NOT color accuracy or illuminant angles. '
        'The same image-mean pooled objective is preserved by group-size weighting. Three shuffled partitions '
        'retain every group size and image. Equal-pair cosine measures gradient agreement only.',
        '', '| Frozen seed17 model | True-person cosine | Shuffled cosine range | True negative pairs | Pooled color/mode cosine | Weighted auxiliary/color norm |',
        '|---|---:|---:|---:|---:|---:|']
    for m in data['models']:
        color=[c for c in m['cases'] if c['objective']=='color'];real=color[0]['statistics'];random=[c['statistics']['mean_cosine'] for c in color[1:]]
        s={'model':m['model'],'true_cosine':real['mean_cosine'],'shuffle_min':min(random),'shuffle_max':max(random),
           'shuffle_mean':float(np.mean(random)),'negative_pair_fraction':real['negative_pair_fraction'],
           'auxiliary_cosine':m['pooled_color_mode_cosine'],'weighted_auxiliary_norm_ratio':m['weighted_aux_to_color_norm']}
        summaries.append(s)
        lines.append(f"| {m['model']} | {s['true_cosine']:.4f} | {s['shuffle_min']:.4f} to{s['shuffle_max']:.4f} | {100*s['negative_pair_fraction']:.2f}% | {s['auxiliary_cosine']:.4f} | {s['weighted_auxiliary_norm_ratio']:.4f} |")
        for case in m['cases']:
            for step in case['steps']:steps.append({'model':m['model'],'partition':case['partition'],'objective':case['objective'],**step})
    own_rise=sum(s['own_objective_change']>0 for s in steps);skin_rise=sum(s['skin_delta_e00_change_image_mean']>0 for s in steps)
    lines += ['', 'Mixed-camera models show less agreement between true-person gradients than between shuffled groups. '
        'The single-camera models do not show this consistent separation. Mixed-camera shuffling also mixes cameras, '
        'capture conditions and skin-color distributions; it cannot isolate harmful person shortcuts. '
        'Legitimate label heterogeneity and cancellation near pooled stationary points remain counterexamples.',
        '', 'All six full-TRAIN pooled color/mode cosines are positive; weighted auxiliary norm ratios range '
        f"{min(s['weighted_auxiliary_norm_ratio'] for s in summaries):.4f} to{max(s['weighted_auxiliary_norm_ratio'] for s in summaries):.4f}. "
        'This is not a causal intervention on the auxiliary task throughout training. It neither proves the task '
        'necessary nor supports blaming it for current generalization failure. Earlier minibatch/teacher diagnostics '
        'used different checkpoints and averaging, so their cosines are not directly comparable.',
        '', '## Actual skin color under finite TRAIN interventions',
        '', 'Every step starts from the frozen state and is immediately undone. Step sizes are absolute parameter '
        'L2 lengths, not AdamW learning rates. Three group indices and two lengths were fixed before observations. '
        'The derivative objective is MSE or MSE+0.1CE; actual instrument-reference DeltaE00 is separately recomputed.',
        '', f'{len(steps)-own_rise}/{len(steps)} steps reduce their own group objective; '
        f'{skin_rise}/{len(steps)} increase full-TRAIN mean DeltaE00. These correlated interventions are not '
        'independent samples, a significance test, or new generalization performance.',
        '', '| Parameter step length | Maximum first-order remainder | Median relative remainder | Mean skin DeltaE00 change range |',
        '|---:|---:|---:|---:|']
    finite=[]
    for length in (.0001,.001):
        ss=[s for s in steps if s['length']==length]
        row={'length':length,'max_remainder':max(s['max_first_order_remainder'] for s in ss),
             'median_relative_remainder':float(np.median([s['max_first_order_remainder']/s['max_first_order_magnitude'] for s in ss])),
             'min_delta_e00_change':min(s['skin_delta_e00_change_image_mean'] for s in ss),
             'max_delta_e00_change':max(s['skin_delta_e00_change_image_mean'] for s in ss)}
        finite.append(row);lines.append(f"| {length} | {row['max_remainder']:.8g} | {row['median_relative_remainder']:.6f} | {row['min_delta_e00_change']:+.6f} to{row['max_delta_e00_change']:+.6f} |")
    lines += ['', 'Relative remainder is max absolute group remainder divided by max first-order magnitude within '
        'that intervention. It is not relative color error. Local first-order predictions become less precise for '
        'the larger step; amax and other piecewise operations do not guarantee globally smooth behavior.',
        '', '![Diagnostic controls and TRAIN color changes](gradient_transfer.png)',
        '', '## Verification and compute scope',
        '', f"Audit PASS:6 independently recomputed pooled gradients,12 recomputed group Gram matrices using batch17 "
        f"instead of32,288 intervention arithmetic checks,12 independently implemented finite-update replays, "
        f"{audit['scalar_color_cases']:,} scalar CIEDE2000 cases. Maximum scalar gap "
        f"{audit['max_scalar_delta_e00_gap']:.3g}; gradient/Gram gap{audit['max_gradient_or_gram_gap']:.3g}; "
        f"finite replay gap{audit['max_finite_replay_gap']:.3g}. All original checkpoint hashes remain unchanged.",
        '', f"The run used FP64 copies for derivative diagnosis, not deployed precision. Total diagnostic computation "
        f"{sum(m['diagnostic_seconds'] for m in data['models']):.2f}s; maximum allocated GPU memory "
        f"{max(m['peak_allocated_mib'] for m in data['models']):.2f}MiB on RTX4060. These are diagnostic runtime/memory, "
        'not training VRAM or batch1 inference latency. No new ONNX/TensorRT/export result.',
        '', '## Decision and limits for Luma',
        '', 'Do not infer universal harmful gradient conflict or prioritize a large gradient-invariance sweep from '
        'these snapshots. Removing auxiliary mode loss or direct squaredDeltaE00 optimization has already been '
        'tested in earlier source experiments; do not relabel those as new mechanisms.',
        '', 'Gradient matching is existing prior art: [Fish](https://arxiv.org/abs/2104.09937) approximates an '
        'inter-domain gradient-inner-product objective; [Fishr](https://proceedings.mlr.press/v162/rame22a.html) '
        'matches gradient variances. This diagnostic reproduces neither published method or author scores. '
        'No author code or weights were adopted.',
        '', 'Independent skin evidence is unchanged: primary mean4.4570/80%4.1591 versus ordinary '
        'fusion4.3005/4.1447 DeltaE00. No strongest-baseline win, facial-phone validation or camera independence. '
        'The next decision is documented separately; the unbounded innovation goal remains active and unmet.']
    with (OUT/'finite_steps.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(steps[0]));writer.writeheader();writer.writerows(steps)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained');xs=np.arange(6)
    axes[0].scatter([s['true_cosine'] for s in summaries],xs,label='True people',color='#b03b3b',zorder=3)
    for j,s in enumerate(summaries):axes[0].plot([s['shuffle_min'],s['shuffle_max']],[j,j],color='#276e99',lw=5,label='Three size-matched shuffles' if j==0 else None)
    axes[0].axvline(0,color='.7',lw=1);axes[0].set_yticks(xs,[s['model'].replace('__',' / ').replace('person_color','combined') for s in summaries],fontsize=8)
    axes[0].set_xlabel('Mean gradient cosine (not color accuracy)');axes[0].set_title('No consistent single-camera separation');axes[0].legend(fontsize=8)
    for j,objective in enumerate(('color','joint')):
        a=[s['skin_delta_e00_change_image_mean'] for s in steps if s['length']==.001 and s['objective']==objective]
        axes[1].scatter(a,np.arange(len(a)),s=13,alpha=.65,label=objective)
    axes[1].axvline(0,color='.5');axes[1].set_xlabel('Change in full-TRAIN mean skin DeltaE00');axes[1].set_ylabel('Fixed intervention index (not independent trials)')
    axes[1].set_title('A successful local step can worsen skin error');axes[1].legend()
    fig.suptitle('TRAIN mechanism diagnostic only — no new held-out accuracy',fontsize=12)
    fig.savefig(OUT/'gradient_transfer.png',dpi=160);plt.close(fig)
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(OUT/'summary.json',{'models':summaries,'finite_step_summary':finite,'own_objective_increases':own_rise,
        'pooled_skin_error_increases':skin_rise,'steps':len(steps),'train_only':True,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'results.json',OUT/'audit.json']}})
    print(json.dumps({'report':str(OUT/'report.md'),'steps':len(steps),'skin_error_increases':skin_rise}))


if __name__=='__main__':main()
