# Inverting paired consistency: common skin-color bias objective

Frozen after source correspondence diagnostic and before these fits. Hypothesis:
reducing the relative penalty on paired-view deviations may allocate capacity
to shared skin-reference bias. This inverts previous output-consistency losses.
This is a controlled objective experiment, not an asserted novel architecture.

Keep CaptureColor('mixture'), 929,297 stored parameters, unchanged 128-pixel
source cache / 64x18 patch inputs, AdamW/cosine 80 epochs and three seeds
17/29/43. Use the exact existing site sampler: 16 sites with two genuine views
per step, first-view half then second-view half. Exact equal site references
are required; singleton sites may repeat the same view as in the original.
No camera ID, paired photo, mode or reference is needed at inference.
Existing capture-mode cross-entropy auxiliary loss remains 0.1 in every arm.

Let I be ordinary mean squared standardized-Lab error of individual predictions,
G the mean squared error of each two-view mean against its common reference,
and D = mean squared paired prediction difference / 4. Algebraically I=G+D.
Four arms with no hyperparameter search:

- individual: I, exact historical mixture-MSE control;
- consistency: I+D = G+2D, increases view agreement;
- shared_half: 0.5I+0.5G = G+0.5D, weakens the agreement penalty;
- shared_only: G, deletes the individual deviation penalty.

Failure mode: large opposite single-image errors cancel under G. Evaluate ONLY
single-image output as the model result. Pair-mean accuracy is a diagnostic
and must never replace single-image accuracy in the result table.

All 36 fits: four arms x three seeds x mixed / SLR-to-iPod / iPod-to-SLR source
protocols. TRAIN-only target scales. Epoch selected using native-Lab DeltaE00
on the same-camera source validation people, then source other-camera cohort
evaluated. Different cameras have different people; not pure sensor causality.
All source validation camera families have been explored before. Exposed
independent MSKCC TEST and CAL, and UMINHO held-outs, remain untouched.

Record all seeds, histories, best/final checkpoints and predictions, mode and
capture strata, train+selection CUDA allocation and parameter count. Evaluate
100/95/90/80/70/60% coverage using the same uncalibrated four-hypothesis RMS
Lab disagreement for every arm, plus the full curve. This is not calibrated
expected DeltaE00. No latency optimization/export unless positive evidence.
Compare strong historical plain-MSE and mixture-MSE in all three protocols.
Demand benefits beyond a single direction before promoting the mechanism.
