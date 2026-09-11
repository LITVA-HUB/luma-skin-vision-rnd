# Post-hoc TRAIN-only objective-gradient diagnostic

Motivation observed after local-teacher mixed fitting: capture-mode accuracy
improved while native skin-color accuracy worsened. This diagnostic is
post-hoc, not a predeclared superiority test and not evidence of causality.
No parameter updates or new model selection. Examine the 12 completed mixed
checkpoints (all four arms, all three seeds), using only original TRAIN inputs
and targets. No source validation images or reserved endpoints loaded here.

Use four32-image batches from the existing paired-site sampler:64pairs drawn
using NumPy default_rng(81017),16pairs perbatch. Same draw for all checkpoints.
Compute standardized-Lab MSE gradient and already-weighted .1capture-mode CE
gradient, separately, for local RGB encoder, adapter, context, votes, gate and
all parameters. Record norms, dot products, cosine where both norms are nonzero,
and primary-loss directional derivative along negative auxiliary gradient.
Missing gradients are zero; zero-norm cosine is undefined, not zero agreement.
CPU float32 forward/autograd, FP64 norm/dot aggregation, two CPU threads.

Validate checkpoint hashes and unchanged state before/after, plus no populated
parameter .grad or optimizer. A negative gradient dot means an infinitesimal
auxiliary-descent step can increase primary loss locally. It does not prove
the auxiliary objective caused generalization harm, or that dropping it will
help. Four in-sample batches at selected checkpoints cannot establish training
trajectory-wide conflict or an accuracy gain. Preserve every arm/batch/block.
