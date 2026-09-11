# Frozen source known/unseen camera transfer after local-reference screen

The preceding TRAIN-only LOPO screen and independent audit are complete.
Color-affine mean4.58597 versus global-ridge5.42585, p9510.01571 versus11.43643,
21/24 people improved; exact80%4.20308 versus4.98987. This is an exploratory
source signal, not novelty or independent phone accuracy. This follow-up is
fixed before decoding its new output arrays; do not tune the six methods.

## Exact continuation without changes to the methods

Reuse the locked skin_local_reference.py and all six systems unchanged.
color36 inputs, bank-only scales, ridge alpha1, Gaussian appearance bandwidth1,
predicted-color CIEDE2000 bandwidth5, weights sum N, no clipping/metadata.
No new choices based on results, no fit/selection on source validation labels.

Three training banks:
- mixed: all24TRAIN people/966images; evaluate6VALIDATION people/264images.
- from_SLR:8TRAIN people/323images; knownSLR3VALIDATION people/132images and
  unseen iPod3people/132images.
- from_ipod:16TRAIN people/643images; knowniPod3VALIDATION people/132images and
  unseen SLR3people/132images.

All source TRAIN/VALIDATION people/sites are disjoint. Unseen evaluation device
is completely absent from the training bank. People and devices are confounded;
this is observational transfer, not paired causal camera calibration. Source
validation has been extensively reused historically, so it is NOT a new final
test. Original independent CAL/TEST and other held-outs remain unread.

Compute all264 query predictions once per bank:3global fits,1584 nonuniform
local affine solves,1584 weighted means,792 uniform-limit checks. Report30
method/domain evaluations,4752 per-image method cases. All methods retained.

## Measures and comparators

Same native-Lab DeltaE00 metrics and shared descriptor-novelty ordering as the
preceding screen. Within each bank/domain, every system accepts the same
images at100/95/90/80/70/60%. Full curves and catastrophic>10 tails retained.
This is not a calibrated C+ or expected-error head. No threshold selected.

Compare ordinary global ridge, local means and affine systems directly. Strong
previously locally reproduced compact neural models are contextual comparators
on the same source people; cite their exact reports and training/selection
differences. Do not call them capacity/budget-matched to this linear bank or
reuse their coverage numbers under a different ranking. No author numbers
inserted as reproduced results. No independent-test accuracy updated.

## Verification and decision

Bind prior screen/audit, method code, new runner/protocol and both source cache
hashes before running. Independently replay all weighted solves and Gaussian
weights; check bank/scaler/device exclusion, unchanged methods, scalar native
color errors and shared coverage. Bank images/targets and predictions stay
private in ignored experiments. Only aggregate evidence committed.

A win over ridge alone does not establish a new universal method. Assess both
unseen directions and strong neural comparators. A partial useful signal can
motivate a separately frozen iterative/reference or learned-affinity experiment;
do not select extra iteration counts on these results. No export/optimization
until useful accuracy evidence warrants the added engineering.
