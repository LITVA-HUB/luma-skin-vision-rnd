# Frozen spatially valid graph/support combination

Actual MSKCC instrument-native skin Lab / DeltaE00. TRAIN/source VALIDATION
only, known exposed independent TEST/CAL archive and all external held-outs
remain untouched. Original MSKCC CC-BY. No new data, external weights or cloud.

## Structural assumption and combination

The existing graph connects an 8x8 grid of actual image patches. Same-site
paired images are not verified as pixel registered; mixed bags must not be
passed through spatial edges as if their boundaries were measured neighbors.

Use the same SpatialColor graph3 stored modules and initialization in all arms.
At inference always disable graph processing and use the shared plain color
core. During training perform two forward passes per batch of 32 original
images (16 same-site pairs):

- plain_raw: plain original + plain original.
- plain_paired: plain original + plain paired_stratified bag.
- graph_raw: graph original + plain original.
- graph_paired: graph original + plain paired_stratified bag.

Loss is 0.5 standardized-Lab MSE for each output. No mode auxiliary loss, camera
ID input, consistency/distillation target or inference-time adaptation. Both
outputs use the actual common site reference. Graph original input is never
augmented. Every arm executes the same two-pass structure; graph computation
is extra in its two arms and is reported, not hidden as equal compute.
The second pass uses the pre-existing fixed observed-patch plan streams and
same-site target checks. A derived bag is not a new photographed observation.

## Budget and evidence

Four arms x three seeds17/29/43 x three protocols mixed/from_SLR/from_ipod =36
fits, all run regardless of early outcomes. Same80epochs,AdamW0.001,wd0.01,
cosine0.00001, source site-pair sampling and separate augmentation RNG. Target
scales fit only TRAIN. Same-camera validation patient-mean DeltaE00 selects epoch;
other-camera population is evaluated afterward. All source data is reused, and
camera protocols change people as well as device. No new independent claim.

All arms store the same modules; only the plain core is active at inference.
Compare against the archived original graph_always and plain baselines, while
noting that this experiment has an additional supervised plain pass. It tests
a combination and a changed training schedule, not an isolated graph mechanism.

Full mean/median/p95/tails and 100/95/90/80/70/60% curves use the shared
TRAIN-only input novelty rank from the expert experiment. This is a common
accept-set accuracy comparison, not a calibrated C+ or error bound.
Exact initialization, source plan counts, original graph inputs, model replay,
scalar CIEDE2000, source-only selection and inference branch removal are audited.
No export/optimization unless stronger positive evidence exists.

Failure modes: double objectives may weaken either useful regularizer, original
graph bias may remain capture-specific, or the virtual support may harm known
conditions. Successful one-way transfer is not universal facial phone accuracy.
Training-only branches, graph smoothing and observed-patch mixing are established
ideas; combining these partial leads does not itself establish novelty.
