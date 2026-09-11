# Decision after paired-capture invariance

All27fits and81prediction-array replays are complete. Previous independent skin
test evidence remains unchanged. Source experiments were useful progress:
they falsify specific mechanisms and change the next action, not just restate
the problem. No active training handle remains at this checkpoint.

## Rejected assumptions

1. Lower variation across photographs necessarily means more accurate color.
   Output consistency reduced mixed-source capture disagreement2.7488to2.5056
   while worsening instrument error3.4855to3.5123. Both are DeltaE00 but compare
   different quantities. Predicting a constant is a counterexample to using
   repeatability alone as success.
2. Dominant within-site feature changes can safely be removed. The hard
   three-direction quotient scores4.1101 versus standardized3.6091 and raw3.4855.
   Removing capture variation can remove information needed to infer color.
3. Agreement regularization necessarily transfers to another camera. Every
   seed worsens in both source-only camera directions. This is diagnostic
   exploratory evidence; architecture selection already saw both source cameras.

## Next alternatives to implement and falsify

A. Capture-aware latent conditioning. Retain process information, infer it
internally, and use it to select a correction. Training-only four-mode metadata
is available without adding camera ID at inference. Strong control: same
backbone and auxiliary mode supervision, ordinary direct regression. Candidate:
mode-conditioned color hypotheses with learned single-image gate. Mechanism:
disambiguate acquisition-dependent appearance instead of making representation
blind to it. Assumption: process is visually recognizable and its effect
transfers. Failure: the gate learns device/patient shortcuts or process labels
encode non-transferable settings. Cheap experiment: fixed3seed source fits and
source-camera-held-out refits with no evaluation-side calibration. Do not call
an ordinary mixture new; isolate the contribution from auxiliary supervision.

B. Correct the objective geometry. Current optimization uses standardized-Lab
MSE, evaluation uses CIEDE2000. Implement a numerically verified differentiable
squared CIEDE2000 loss with Sharma fixtures and finite-gradient/zero-color tests.
Compare matched raw/color-condition models under both losses. Failure: gradients
near hue boundaries, optimization instability, or lower average error with
worse tails. This is a standard loss control, not a novelty claim.

C. Material/process factorization with reconstruction. A small latent nuisance
channel may explain image evidence while a separate instrument-anchored color
channel stays useful. Assumption: training captures constrain factorization.
Failure: nuisance carries all information, color channel absorbs process, or
reconstruction rewards irrelevant texture. Start only after A/B establish a
strong matched control; do not spend on a complex decoder before cheap falsifiers.

All new development stays on source roles. A new confirmatory generalization
claim requires a distinct legitimate instrument-referenced cohort; the old
TEST is exposed and cannot be recycled as fresh evidence. Ordinary phone facial
accuracy, <=2median/<=5p95 at80% acceptance, novelty and commercial product
validation remain unachieved. No proprietary data collection is proposed.
