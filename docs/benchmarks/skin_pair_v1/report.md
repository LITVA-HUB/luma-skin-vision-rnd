# Paired-capture invariance: direct skin-color experiment

REAL original MSKCC skin images and actual instrument Lab references; all numbers below are REPRODUCED LOCALLY. This experiment uses only the original TRAIN/VALIDATION roles. The previously exposed independent400-image test and208-image calibration data were not loaded. Source validation has been studied before; it is not fresh confirmation.

## Mechanisms tested

Raw: same compact PatchVotes backbone with site-balanced paired sampling. Standardized: same model with TRAIN channel scaling. Quotient3: remove the top three within-site feature-variation directions before the network. Output: add same-site predicted-color consistency. VICReg: additionally align projected context while preventing representation collapse. All have single-image inference with no camera/site/mode input; paired captures are used only during training.

All27fits use seeds17/29/43,80epochs, identical within-protocol backbone initialization, paired sampling and optimization budgets. Inference backbone924,932parameters; VICReg adds32,832training-only projector parameters. The model terms are existing approaches or adaptations, not evidence of patent novelty.

## Mixed-camera source results

24TRAIN people /966images;6source-validation people /264images. Values averaged across three separate seed models are not the accuracy of an ensemble. Lower reference error and lower capture disagreement are distinct goals.

| Arm | Mean DeltaE00, seed17 /29 /43 | Mean over seeds | Same-site capture disagreement |
|---|---|---:|---:|
| raw | 3.4851 / 3.5127 / 3.4587 | 3.4855 | 2.7488 |
| standardized | 3.5806 / 3.6027 / 3.6441 | 3.6091 | 2.9142 |
| quotient3 | 4.0912 / 4.1960 / 4.0431 | 4.1101 | 3.0339 |
| output | 3.5188 / 3.4939 / 3.5242 | 3.5123 | 2.5056 |
| vicreg | 3.5507 / 3.5174 / 3.5668 | 3.5450 | 2.5573 |

The output-consistency challenger was selected by the predeclared lowest three-seed source patient-mean rule among non-baseline arms, even though it did not beat the raw baseline overall. The same-site disagreement decrease is not a skin-accuracy victory. The hard quotient loses color accuracy substantially; this is consistent with capture-sensitive feature directions also containing useful color information, but does not isolate the cause from optimization/representation effects.

## Camera-held-out fitting on source cohorts

Fresh weights and all feature/target statistics fit one camera only. Epoch selection sees only three source-validation people with that same camera. The opposite camera is evaluated only after checkpoint selection. Challenger architecture was chosen on the earlier mixed-camera source screen, so this is exploratory and cannot be claimed to have no held-out-camera exposure at the research-design level.

| Training camera -> evaluation | Arm | TRAIN /selection /evaluation people | Evaluation images | Mean DeltaE00, seed17 /29 /43 | Mean over seeds |
|---|---|---|---:|---|---:|
| SLR -> iPod | output | 8 /3 /3 | 132 | 6.1419 / 6.3260 / 5.9898 | 6.1526 |
| SLR -> iPod | raw | 8 /3 /3 | 132 | 5.3482 / 5.4913 / 5.9794 | 5.6063 |
| iPod -> SLR | output | 16 /3 /3 | 132 | 6.3551 / 6.5163 / 6.8254 | 6.5656 |
| iPod -> SLR | raw | 16 /3 /3 | 132 | 5.8650 / 6.3605 / 6.1943 | 6.1399 |

from_SLR: challenger minus raw mean across seeds = **+0.5463 DeltaE00** (negative favors challenger).

from_ipod: challenger minus raw mean across seeds = **+0.4257 DeltaE00** (negative favors challenger).

Only three evaluation people per camera and prior source exposure limit inference. Cross-camera direction also changes person populations and amount of training data. No ordinary iPhone/Android facial-selfie accuracy follows. These measurements must not overwrite or be pooled with the prior independent skin test.

## Integrity and next R&D decision

Independent audit: 27fits, 81exact prediction-array replays, 5544scalar CIEDE2000 comparisons, maxgap5.33e-15. All target scales/quotients recomputed from the fitting people only; matched backbone initialization verified. No TEST/CALIBRATION endpoints loaded. No new latency/ONNX/TensorRT claim; optimizing deployment is not justified by these source-only results.

Working inference: unconditional capture invariance is not enough. Do not escalate a repeatability improvement into color accuracy or camera independence. Preserve these negative and mixed results. Next test the opposite mechanism: retain capture-process evidence and condition the color estimate on an internally inferred process, with auxiliary capture-mode labels used only during training. Compare against a same-capacity auxiliary-task baseline so a mixture mechanism cannot win merely from additional supervision. Separately test whether training directly in perceptual color-error geometry improves over standardized-LabMSE; that objective change alone is not novel.

[Frozen protocol and fresh primary prior-art sources](../../research/skin_pair_invariance_protocol_v1.md). [Previous independent400-image benchmark](../skin_mskcc_selective_v1/report.md) remains the authoritative independent endpoint. The innovation goal remains active and unachieved.
