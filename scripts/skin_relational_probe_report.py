"""Relational signal versus actual skin accuracy, with scope-separated controls."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skin_relational_probe_run import OUT
from skin_mskcc_data import ROOT,sha
from skin_pair_train import write


def main():
    r=json.loads((OUT/'results.json').read_bytes());audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    for p,h in audit['bindings'].items():assert sha(ROOT/p)==h
    lines=['# TRAIN relational compatibility falsifier',
        '', 'This is a source-mechanism probe, not a new independent color-accuracy result. '
        'Real original MSKCC CC-BY skin images and native instrument Lab; original TRAIN only. '
        'No source VALIDATION, CAL, TEST or additional held-out dataset was loaded.',
        '', '## Support and controls',
        '', '24 people,966 images,248 measured sites;1,421 same-site capture pairs and18,229 '
        'within-person between-site pairs. Every same-site pair has a wrong-site control from '
        'the same person, matching the second image capture mode and image type. There are '
        '**zero same-site cross-camera pairs**. Anatomical site identity does not establish '
        'pixel registration or identical imaged tissue.',
        '', 'Same-site reference DeltaE00 is zero by construction. Closest wrong-site reference '
        'distances range0.2803 to11.8487, median2.4103. They do not exactly match the positive '
        'color-distance distribution. Of1,421 pairs,44/209/607/1,319 have a nearest eligible '
        'control within0.5/1/2/5 DeltaE00, from5/14/22/24 people. The closest-color control '
        'uses reference labels only for diagnostic matching; it is not an inference rule.',
        '', 'The uniform and closest-color control lists contain1,375 and1,222 unique '
        'unordered negative pairs. Repeated images/pairs are retained and disclosed. '
        'Large pair counts do not create more independent people.',
        '', '## Does the representation distinguish the same site while preserving color?',
        '', 'Preference is the equal-person mean of the fraction for which the same-site '
        'pair has smaller descriptor distance than its matched wrong-site pair; ties count0.5. '
        'It is NOT skin-color accuracy. Correlation is equal-person Spearman association '
        'between representation distance and actual between-site reference DeltaE00. '
        'Arbitrary feature RMS distances are never labelled DeltaE00.',
        '', '| Representation | Scope | Closest-control preference | Control within1 DeltaE00 | Control within2 DeltaE00 | True-color rank association |',
        '|---|---|---:|---:|---:|---:|']
    summary=[];csvrows=[]
    for m in r['representations']:
        near={c['maximum_reference_delta_e00']:c for c in m['comparisons'] if c['control']=='near'}
        corr=m['between_color_person_mean_spearman'];scope=m['scope']
        short='TRAIN encoder, descriptive' if scope.startswith('descriptive') else 'LOPO linear/mean' if scope.startswith('excluded') else 'unfitted descriptor'
        lines.append(f"| {m['representation']} | {short} | {100*near[None]['person_mean_preference']:.2f}% | {100*near[1.]['person_mean_preference']:.2f}% | {100*near[2.]['person_mean_preference']:.2f}% | {corr:.4f} |" if corr is not None else f"| {m['representation']} | {short} | 50.00% | 50.00% | 50.00% | undefined (constant) |")
        summary.append({'representation':m['representation'],'scope':scope,'nearest_preference':near[None]['person_mean_preference'],
            'within1_preference':near[1.]['person_mean_preference'],'within2_preference':near[2.]['person_mean_preference'],'rank_association':corr})
        for c in m['comparisons']:csvrows.append({'representation':m['representation'],'scope':scope,**c})
    lines += ['', 'The frozen learned contexts were trained on all original TRAIN people, including '
        'those whose pairs are analysed. Excluding a person from feature standardization does not '
        'remove encoder label exposure. Their apparent advantage is descriptive only. The48 '
        'linear ridge fits exclude the entire evaluated person, including all their sites, '
        'from every fit and normalization. Original TRAIN has still been reused across R&D: '
        'these folds are exploratory cross-validation, not a new confirmatory cohort.',
        '', 'Ridge36 gives70.08% closest-control preference and0.3955 color association; '
        'the all-TRAIN learned context gives70.92% and0.3686. For controls within1 DeltaE00, '
        'ridge36 gives72.64% versus context61.07%, on209 dependent pairs from14 people. '
        'Thus this probe does not establish a special comparative representation advantage '
        'beyond a simple color predictor. It also does not prove that every relational model '
        'must fail. Texture/site recognition and color differences remain potential explanations.',
        '', '## Actual direct skin-color error of the excluded-person controls',
        '', 'Each of24 people is predicted by a ridge model fitted on the other23. '
        'All966 predictions are scored against genuine native instrument Lab. '
        'No camera identity or capture metadata enters these regressions.',
        '', '| LOPO control | Mean DeltaE00 | Median | p95 | Equal-person mean |',
        '|---|---:|---:|---:|---:|']
    for m in r['representations']:
        if 'absolute_color_error' in m:
            a=m['absolute_color_error'];lines.append(f"| {m['representation']} | {a['mean']:.4f} | {a['median']:.4f} | {a['p95']:.4f} | {a['person_mean']:.4f} |")
    lines += ['', 'These numbers cannot be ranked against the separate400-image independent test '
        'as if the population/protocol were the same. No new risk-coverage or unseen-camera '
        'result is established. Good relative pair ranking is insufficient for accurate '
        'absolute skin color: ridge36 still has mean5.4259 and p9511.4364 DeltaE00 here.',
        '', '## Mathematical limit and prior art',
        '', 'For additive comparison d(x,a)=f(x)-f(a), uniformly averaging y(a)+d(x,a) '
        'is exactly f(x)+mean(y(a)-f(a)). Repeated comparisons only add a constant offset. '
        'Complete antisymmetric zero-cycle comparisons recover an ordinary potential up to '
        'a constant. The actual frozen predictor identity checks differ by at most '
        f"{r['additive_identity_max_gap']:.3g} and{r['cycle_potential_max_gap']:.3g}. "
        'Tests include a pure cyclic counterexample; this is algebra, not synthetic accuracy.',
        '', '[Relation Networks, CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Sung_Learning_to_Compare_CVPR_2018_paper.html) '
        'already learn comparison metrics from episodes. '
        '[Deep Kernel Learning, AISTATS2016](https://proceedings.mlr.press/v51/wilson16.html) '
        'combines learned representations and kernels. '
        '[HodgeRank](https://arxiv.org/abs/0811.1067) separates potential and cyclic comparison components. '
        'These are prior art, not our inventions or locally reproduced author methods. '
        'No third-party implementation or weights were adopted.',
        '', '## Verification and next decision',
        '', f"Audit PASS: {audit['candidate_checks']:,} independent candidate checks,48 augmented "
        f"least-squares refits,4 held-target perturbations,2 context replays,100 preference checks, "
        f"250 rank checks and{audit['scalar_color_cases']:,} scalar color cases. Maximum scalar "
        f"gap{audit['max_scalar_gap']:.3g}; coefficient gap{audit['max_ridge_coefficient_gap']:.3g}. "
        'All original neural checkpoints and cache hashes remain unchanged.',
        '', 'Do not promote another pair network from same-site recognition alone. A narrower '
        'next test is ordinary query-dependent reference weighting, measured directly by '
        'excluded-person absolute skin DeltaE00 with global-ridge and weighted-mean controls. '
        'This is a standard local-regression comparator, not novelty. It tests whether '
        'compatibility weighting contributes beyond the degenerate constant offset before '
        'spending on a learned pair mechanism.',
        '', 'Independent MSKCC primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447 '
        'is unchanged. No strongest independent-baseline win, ordinary-phone facial '
        'validation, novel compact architecture or calibrated refusal guarantee. '
        'No new model VRAM/latency/export claim. Goal active and unmet.',
        '', '![Comparative signal and actual color error](relational_probe.png)']
    with (OUT/'comparisons.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for j,s in enumerate(summary):
        if s['rank_association'] is None:continue
        color='#bd6655' if s['scope'].startswith('descriptive') else '#267398' if s['scope'].startswith('excluded') else '.5'
        axes[0].scatter(100*s['within1_preference'],s['rank_association'],color=color,s=45)
        offsets={'frozen_lab_image':(6,9),'frozen_lab_combined':(6,-15),'context_image':(6,10),'context_combined':(6,-14),'patch54':(6,8)}
        axes[0].annotate(s['representation'],(100*s['within1_preference'],s['rank_association']),xytext=offsets.get(s['representation'],(4,5 if j%2 else -13)),textcoords='offset points',fontsize=8)
    axes[0].set_xlabel('Within1 DeltaE00 controls: preference (%)');axes[0].set_ylabel('Within-person true-color rank association')
    axes[0].set_title('Red: encoder saw all TRAIN labels\nBlue: person-excluded linear fits; gray: descriptors',fontsize=10);axes[0].grid(alpha=.15)
    axes[0].set_xlim(55,78);axes[0].set_ylim(.19,.53)
    ms=[m for m in r['representations'] if 'absolute_color_error' in m]
    axes[1].bar([m['representation'] for m in ms],[m['absolute_color_error']['mean'] for m in ms],color=['#488dad','#267398','.7'])
    axes[1].set_ylabel('Actual mean skin DeltaE00 (lower is better)');axes[1].set_title('Exploratory TRAIN leave-one-person-out\nDifferent cohort from independent400-image test',fontsize=10)
    for j,m in enumerate(ms):axes[1].text(j,m['absolute_color_error']['mean']+.1,f"{m['absolute_color_error']['mean']:.3f}",ha='center')
    axes[1].set_ylim(0,12);fig.suptitle('Relative matching ability does not establish absolute skin-color accuracy',fontsize=12)
    fig.savefig(OUT/'relational_probe.png',dpi=160);plt.close(fig)
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(OUT/'summary.json',{'comparative_results':summary,'absolute_color_error':{m['representation']:m['absolute_color_error'] for m in ms},
        'train_only':True,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),OUT/'results.json',OUT/'audit.json']}})
    print(str(OUT/'report.md'))


if __name__=='__main__':main()
