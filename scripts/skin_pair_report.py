"""Aggregate completed source experiments; cannot train or change selection."""
import json
import numpy as np
from skin_mskcc_data import ROOT,sha
from skin_pair_train import OUT,SEEDS
from skin_pair_invariance import ARMS


def main():
    rows=[json.loads(p.read_bytes()) for p in OUT.glob('*/result.json')];assert len(rows)==27
    audit=json.loads((OUT/'audit.json').read_bytes());assert audit['status']=='PASS'
    choice=json.loads((OUT/'transfer_choice.json').read_bytes());groups={}
    for r in rows:groups.setdefault((r['protocol'],r['arm']),[]).append(r)
    records=[]
    for (protocol,arm),rs in sorted(groups.items()):
        rs=sorted(rs,key=lambda r:r['seed']);assert [r['seed'] for r in rs]==SEEDS
        records.append({'protocol':protocol,'arm':arm,'seeds':SEEDS,
            'individual_mean_delta_e00':[r['full']['mean'] for r in rs],
            'mean_over_seed_mean_delta_e00':float(np.mean([r['full']['mean'] for r in rs])),
            'mean_over_seed_pair_disagreement':float(np.mean([r['paired_repeatability']['mean_delta_e00_between_captures'] for r in rs])),
            'train_people':rs[0]['train_people'],'selection_people':rs[0]['selection_people'],'evaluation_people':rs[0]['evaluation_people'],
            'train_images':rs[0]['train_images'],'selection_images':rs[0]['selection_images'],'evaluation_images':rs[0]['evaluation_images']})
    summary={'scope':'Exploratory source validation and source camera-held-out fitting; not a new independent test',
             'groups':records,'fits':27,'challenger':choice['challenger'],'source_lock_sha256':sha(OUT/'source_lock.json'),
             'all_fixed_rules_in':'docs/research/skin_pair_invariance_protocol_v1.md','reserved_data_read':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    lines=['# Paired-capture invariance: direct skin-color experiment',
      '', 'REAL original MSKCC skin images and actual instrument Lab references; all numbers below are REPRODUCED LOCALLY. This experiment uses only the original TRAIN/VALIDATION roles. The previously exposed independent400-image test and208-image calibration data were not loaded. Source validation has been studied before; it is not fresh confirmation.',
      '', '## Mechanisms tested',
      '', 'Raw: same compact PatchVotes backbone with site-balanced paired sampling. Standardized: same model with TRAIN channel scaling. Quotient3: remove the top three within-site feature-variation directions before the network. Output: add same-site predicted-color consistency. VICReg: additionally align projected context while preventing representation collapse. All have single-image inference with no camera/site/mode input; paired captures are used only during training.',
      '', 'All27fits use seeds17/29/43,80epochs, identical within-protocol backbone initialization, paired sampling and optimization budgets. Inference backbone924,932parameters; VICReg adds32,832training-only projector parameters. The model terms are existing approaches or adaptations, not evidence of patent novelty.',
      '', '## Mixed-camera source results',
      '', '24TRAIN people /966images;6source-validation people /264images. Values averaged across three separate seed models are not the accuracy of an ensemble. Lower reference error and lower capture disagreement are distinct goals.',
      '', '| Arm | Mean DeltaE00, seed17 /29 /43 | Mean over seeds | Same-site capture disagreement |',
      '|---|---|---:|---:|']
    for arm in ARMS:
        a=next(r for r in records if r['protocol']=='mixed' and r['arm']==arm)
        lines.append(f"| {arm} | {' / '.join(f'{x:.4f}' for x in a['individual_mean_delta_e00'])} | {a['mean_over_seed_mean_delta_e00']:.4f} | {a['mean_over_seed_pair_disagreement']:.4f} |")
    lines += ['', 'The output-consistency challenger was selected by the predeclared lowest three-seed source patient-mean rule among non-baseline arms, even though it did not beat the raw baseline overall. The same-site disagreement decrease is not a skin-accuracy victory. The hard quotient loses color accuracy substantially; this is consistent with capture-sensitive feature directions also containing useful color information, but does not isolate the cause from optimization/representation effects.',
      '', '## Camera-held-out fitting on source cohorts',
      '', 'Fresh weights and all feature/target statistics fit one camera only. Epoch selection sees only three source-validation people with that same camera. The opposite camera is evaluated only after checkpoint selection. Challenger architecture was chosen on the earlier mixed-camera source screen, so this is exploratory and cannot be claimed to have no held-out-camera exposure at the research-design level.',
      '', '| Training camera -> evaluation | Arm | TRAIN /selection /evaluation people | Evaluation images | Mean DeltaE00, seed17 /29 /43 | Mean over seeds |',
      '|---|---|---|---:|---|---:|']
    for a in records:
        if a['protocol']=='mixed':continue
        label='SLR -> iPod' if a['protocol']=='from_SLR' else 'iPod -> SLR'
        lines.append(f"| {label} | {a['arm']} | {a['train_people']} /{a['selection_people']} /{a['evaluation_people']} | {a['evaluation_images']} | {' / '.join(f'{x:.4f}' for x in a['individual_mean_delta_e00'])} | {a['mean_over_seed_mean_delta_e00']:.4f} |")
    for protocol in ['from_SLR','from_ipod']:
        a=next(r for r in records if r['protocol']==protocol and r['arm']=='raw')
        b=next(r for r in records if r['protocol']==protocol and r['arm']==choice['challenger'])
        lines += ['', f"{protocol}: challenger minus raw mean across seeds = **{b['mean_over_seed_mean_delta_e00']-a['mean_over_seed_mean_delta_e00']:+.4f} DeltaE00** (negative favors challenger)."]
    lines += ['', 'Only three evaluation people per camera and prior source exposure limit inference. Cross-camera direction also changes person populations and amount of training data. No ordinary iPhone/Android facial-selfie accuracy follows. These measurements must not overwrite or be pooled with the prior independent skin test.',
      '', '## Integrity and next R&D decision',
      '', f"Independent audit: {audit['fits']}fits, {audit['exact_prediction_array_replays']}exact prediction-array replays, {audit['independent_scalar_delta_e00_comparisons']}scalar CIEDE2000 comparisons, maxgap{audit['maximum_metric_gap']:.3g}. All target scales/quotients recomputed from the fitting people only; matched backbone initialization verified. No TEST/CALIBRATION endpoints loaded. No new latency/ONNX/TensorRT claim; optimizing deployment is not justified by these source-only results.",
      '', 'Working inference: unconditional capture invariance is not enough. Do not escalate a repeatability improvement into color accuracy or camera independence. Preserve these negative and mixed results. Next test the opposite mechanism: retain capture-process evidence and condition the color estimate on an internally inferred process, with auxiliary capture-mode labels used only during training. Compare against a same-capacity auxiliary-task baseline so a mixture mechanism cannot win merely from additional supervision. Separately test whether training directly in perceptual color-error geometry improves over standardized-LabMSE; that objective change alone is not novel.',
      '', '[Frozen protocol and fresh primary prior-art sources](../../research/skin_pair_invariance_protocol_v1.md). [Previous independent400-image benchmark](../skin_mskcc_selective_v1/report.md) remains the authoritative independent endpoint. The innovation goal remains active and unachieved.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary))


if __name__=='__main__':main()
