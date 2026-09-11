# Frozen TRAIN-person support / representation screen

Real original MSKCC skin images and native instrument Lab; source TRAIN only.
Do not load VALIDATION, CALIBRATION or TEST. Original TRAIN has 8 SLR and 16 iPod
people. A fixed RNG20260911 selects 2/4 people as a six-person internal holdout.
Each draw seed17/29/43 permutes the remaining people separately by camera.
Nested 6/12/18-person training sets retain 1:2 camera proportions. Holdout is
identical across all runs. Counts, role hashes, sites and color ranges recorded.
No identifiers or participant images in public aggregate artifacts.

27 fits: three person counts x three joint subset/model seeds x three arms.
Baseline: original CaptureColor mixture with mode auxiliary CE0.1. Statistics:
zero-initialized residual 18->64->120->18 MLP on actual 18 patch statistics.
Pixels: independently encoded real 16x16 RGB patches, 3->16->24->32 stride-two
3x3 convs, SiLU, global mean, 32->18 zero-output projection added to those same
statistics. No convolution crosses patch boundaries. All share exact initial
downstream core; adapters differ by less than 0.1% total parameters. All use
one image without camera/mode input; acquisition mode is training auxiliary only.
No external weights, invented color targets, paired synthesis or new data.

Fixed 30 rounds of 31 updates, batch32: 930 optimizer steps / 29,760 image draws
per fit, independent of subset size. Uniform image draws with replacement from
selected people, identical draws across arms. AdamW lr0.001 wd0.01, cosine after
each round to0.00001. Standardized native-Lab MSE +0.1 modeCE, same as baseline.
Target scaling uses selected training images only. RGB uint8/255, no augmentation.
FP32 deterministic GPU; no AMP, pretrained model or calibration. Fixed final
checkpoint is primary; retain per-round diagnostics without checkpoint selection.
No comparisons with 80-epoch historical scores as matched experiment results.

Report actual skin DeltaE00 mean/median/p90/p95, patient mean, fractions above5/10.
Compare predictions within the same subset draw. Record nearest TRAIN site-color
distance for actual held-out reference ONLY as a labeled support diagnostic;
never an inference feature or score. Model-independent input-novelty ranking
uses TRAIN mean patch stats, TRAIN std floor1e-6 and nearest Euclidean distance.
Report full curves and fixed100/95/90/80/70/60% coverage. Uncalibrated, not C+.
All three seeds co-vary model initialization and subset ordering: not independent
held-out populations. This small reused TRAIN cohort is an exploratory falsifier,
not a new external test, causal pure-camera effect or a universal phone result.

Hypothesis: learned patch appearance reveals color-relevant information lost by
fixed statistics, particularly as support increases. Counter-hypothesis: it adds
capture shortcuts or overfits the same scarce population. The cheap experiment
is this equal-update learning curve, not a larger network or output-head sweep.
If pixel gains fail against the matched statistics adapter, preserve the negative
and reconsider the representation rather than claim information recovery.
