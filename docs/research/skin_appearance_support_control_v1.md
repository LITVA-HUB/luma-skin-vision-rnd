# Post-hoc empirical support control before interpretation

The 108 appearance/direct models are already fitted on reused source data.
This additional control is frozen before its evaluation; it does not turn
source validation into a fresh confirmatory population. No fitted model is
changed, no reserved endpoint is opened, no external dataset is acquired.

For every direct candidate, transform its predicted Lab without the true image
reference in two ways: nearest actual TRAIN-site Lab atom by CIEDE2000, or
Euclidean projection onto the convex hull of TRAIN-site Lab after per-channel
standardization using those site colors. The hull is an empirical constraint,
not a certified physically realizable skin gamut. The posterior-mean inverse
already lies in this hull; its medoid already belongs to the finite atom set.
This isolates support restriction from the forward-appearance likelihood.

Use existing TRAIN/source VALIDATION caches and direct prediction artifacts.
Apply both controls to all 36 direct models at both selection and evaluation.
For each input/degree/control family select alpha using same-camera validation
patient-mean DeltaE00 and the original tie order. Never select on transfer error.
Common input novelty accepts the same images for all competing predictions.

Nearest-atom indices are checked with independent scalar CIEDE2000. Convex
projection uses a three-dimensional strictly convex quadratic objective with
linear hull inequalities. Check feasibility, nonnegative multipliers and
stationarity at each changed point; interior points remain bitwise identical.
Two unit tests cover an analytic tetrahedron and unchanged observed atoms.

Archive selection/evaluation arrays privately, aggregate all scores/curves,
independent color/coverage checks and unchanged-input fraction. Compare both
controls with the direct and forward candidates, while acknowledging the extra
post-hoc search and repeated source reuse. Do not count a support-only gain as
a contribution from a conditional generative architecture.
