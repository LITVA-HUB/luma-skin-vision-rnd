# Frozen removal versus conditional expert supervision screen

Endpoint: actual MSKCC instrument-native skin Lab / DeltaE00. This experiment
does not measure illuminant angles. TRAIN/source VALIDATION only; exposed
independent TEST/CAL and all reserved external populations remain untouched.
Original MSKCC CC-BY provenance applies. No new external data/code/weights.

## Mechanisms and controls

Eight arms = four mechanisms crossed with raw / paired_stratified training.
Three seeds 17/29/43 and mixed / SLR-to-iPod / iPod-to-SLR = 72 fits. Run all
protocols irrespective of favorable mixed results. Preserve every outcome.

baseline: existing four-hypothesis mixture and 0.1 capture-mode cross entropy.
plain: remove the gate and replace four linear color heads with one. Its initial
color function equals the old uniformly averaged model up to summation precision;
shared backbone initialization is identical. 924,932 versus 929,297 parameters.
No capture-mode auxiliary objective. This removes the whole decomposition;
it is not an isolated gating-only ablation. Compression changes head optimization
geometry and is not an exact refit of the earlier overparameterized plain arm.
uniform_anchor: 0.5 final standardized Lab MSE + 0.5 mean individual-head MSE
+ 0.1 mode CE. Every expert receives the same native color target.
conditional: same coefficients, but individual-head MSE is weighted by the
actual capture-mode label (raw) or observed patch source proportions (paired).
Total primary/anchor color coefficient remains one. Coefficients are fixed,
not selected against the results of this experiment. This is standard supervised
expert specialization, not a claimed novel architecture.

Anchor supervision acts on pooled image hypotheses, not individual patch color
labels. For virtual paired bags the common site Lab is unchanged; source-mode
proportions are derived composition labels, not measured labels of a new photo.
Such bags may not be physically realizable captures. Uniform anchoring controls
extra expert supervision without semantic specialization. No camera/mode ID
is provided at inference and no test-time adaptation is used.

## Matched budget and selection

Same original 64x18 observed tokens, 80 epochs, AdamW 0.001, wd 0.01, cosine to
0.00001, 16 site pairs / 32 bags per step. Same pair and augmentation RNG streams
as the support experiment, independent of model initialization. All source
pairs must share site and exactly matching native references. No cross-camera
same-site pairs exist. Baseline raw uses original hard CE for exact replay.
Same-camera validation patient-mean DeltaE00 selects epoch; other-camera inputs
are evaluated afterward. Camera transfer also changes people/capture composition.
Source validation is extensively reused: all findings are exploratory.

## Accuracy and selectivity

Primary is full mean / median / p95 actual skin DeltaE00, with all three seeds.
Fixed coverages 100/95/90/80/70/60 and full curves use a COMMON input novelty
ranking: nearest TRAIN image Euclidean distance of 18 mean patch statistics,
standardized by TRAIN mean/std (std floor 1e-6). No targets fit this ranking.
Identical accept sets across methods isolate color accuracy at fixed coverage.
This is a simple uncalibrated control, not calibrated C+ or an error guarantee.
Do not compare these curves to previous hypothesis-dispersion rankings as if
the acceptance mechanisms were identical. Conditional-head scoring with true
evaluation mode is a labeled diagnostic only, never the deployable prediction.

Archive exact model/checkpoint replays, source locks, scalar CIEDE2000 checks,
shared initial backbone, original baseline replay, fit VRAM/size/time and all
negatives. No export/optimization or independent accuracy claim from this screen.

## Falsification

Removal may work if specialization is unnecessary or harmful; it may fail if
capture appearance needs conditional mappings. Conditional anchoring may work
if the prior heads lacked useful semantics; it may fail when clinical labels
do not describe unseen-device behavior. Uniform anchoring may explain a gain
through regularization alone. Better mode recognition is not skin color accuracy.
Require gains over strongest matched controls in both transfer directions;
do not promote a one-direction improvement as universal phone performance.
