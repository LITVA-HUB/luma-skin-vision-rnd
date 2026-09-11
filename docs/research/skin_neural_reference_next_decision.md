# Capacity screen decision: preserve the core; attack residual-training mismatch

Previous goal turn was PROGRESS: preserved audited local-reference results and
committed the200,000-parameter allowance. This turn is PROGRESS: implemented
matched heads, froze27 fits, trained all of them and independently audited them.
All jobs completed; no model-training process is left running.

## Actual measured result

All three heads193,795 parameters, total1,123,092. Core929,297 remains exactly
unchanged. Three seeds, three camera protocols,300 final-only updates per head.
Each head sees identical data/draws/initial weights, no inference camera ID.

Native skin DeltaE00, mean of three individual seeds:

| Source protocol | Unchanged core | Ordinary residual | Reference mean | Reference affine |
|---|---:|---:|---:|---:|
| Mixed known | 3.4406 | 3.6198 | 3.5332 | 3.5182 |
| SLR to unseen iPod | 5.0193 | 5.4398 | 5.0618 | 4.9658 |
| iPod to unseen SLR | 5.9635 | 6.4545 | 6.0437 | 6.4782 |

The forward affine gain over its core is about1.07%; mixed/reverse deteriorate.
Strong historical transfer recipes4.8328 and4.9736 still beat these new heads.
The affine method does not consistently beat the matched residual control:
reverse6.4782 is worse than6.4545. Reference averaging is better on reverse,
but also loses the core. Do not combine per-camera winners into a claimed
camera-blind system. No ensemble or after-the-fact selected model is deployed.

All27 heads reduce their full TRAIN standardized MSE, even when bank entries
of the query person are excluded. These are NOT OOF predictions: the frozen
core saw every TRAIN person. This establishes successful optimization and a
generalization failure, not a bug or lack of added parameter capacity.

## Assumption to invert

Current assumption: a head can learn how to repair unseen-person errors from
the core's ordinary training residuals. Invert it: train correction decisions
using queries whose encoder has never seen their person. Excluding the query
person only from memory is insufficient to establish such supervision.
Bank residual optimism and acquisition-specific neighborhoods remain plausible
causes, not proven diagnoses. MSE/DeltaE loss choice alone has already been
tested in earlier capture experiments and is not a new architecture idea.

Next bounded falsifier should isolate this supervision mismatch BEFORE another
large sweep or repeated correction. Use original TRAIN internal person splits,
refit encoder and its scales entirely within the support people, then create
native Lab correction targets for disjoint query people. Preserve a no-head
control and ordinary matched correction head. Source validation remains reused
exploratory evaluation; independent TEST/CAL must stay closed. Never call the
existing full-TRAIN features OOF by merely refitting a final layer.

Important design issue: independently trained encoders can rotate512-dimensional
context coordinates. A cross-fitted head cannot assume these contexts are
interchangeable. Compare a stable representation (predicted native color and
fixed image descriptors) or explicitly verify representation compatibility.
Do not silently mix differently fitted feature/target scalings. Freeze this
choice and the necessary refits before generating correction outcomes.

Alternative worth keeping open: allocate the authorized capacity to learning
the image representation jointly rather than repairing a frozen representation.
That is a different hypothesis, with a newly trained ordinary enlarged-backbone
control. Current failures do not reject it. First inspect prior pixel-adapter
and representation negatives to avoid renaming an already tested mechanism.

The repeated-refinement proposal is not discarded. It needs an update rule
whose training and inference residuals agree; repeatedly adding this fitted
one-step correction is not justified by the current mixed/reverse results.

## Evidence and boundaries

9 core replays,27 independent NumPy heads,2376 augmented solves,36 excluded
reference perturbations,3 exact complete optimizer refits,60 exact arrays,
9504 scalar CIEDE2000 values,360 coverage checks. Learned reference bank fits
within6,137–18,354 extra scalars; complete payload4,529,999–4,578,895 bytes.
Affine batch1 prepared-feature median1.424–3.042ms on RTX4060; excludes JPEG
decode and feature extraction. Cached training peak84.77–110.07MiB includes
resident core and caches, not isolated production VRAM. No export claim.

Original MSKCC CC-BY data and project-authored code. Differentiable solvers,
local regression, learned kernels and residual gates are prior art. Implemented
capacity increase is not an innovation victory. Independent primary4.4570/
80%4.1591 vs ordinary fusion4.3005/4.1447 unchanged. Ordinary facial smartphone
color accuracy remains unvalidated. Goal remains active and unmet.

[Full report](../benchmarks/skin_neural_reference_v1/report.md).
