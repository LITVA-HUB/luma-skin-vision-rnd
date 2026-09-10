# V6: canonical-frame correction evidence — predeclared development screen

This combines two previously useful, separately measured mechanisms: V2
Shades-of-Gray-relative inference for transfer, and V5 physical corrected-color
risk routing for selective source error. Both mechanisms have known prior art.
Their combination is an UNVERIFIED engineering hypothesis, not a novelty claim.
Written while the locked phone benchmark is preparing, before phone model errors.

Hypothesis: removing an image-wide diagonal color frame from BOTH learned scene
context and physical correction evidence improves transfer without losing the
selective advantage of action-conditioned corrected-color evidence. All local
hypotheses and risk queries live in the same residual frame. Predicted illuminants
are transformed back exactly once. A diagonal change is not a full camera/ISP
change; no arbitrary-camera invariance is promised.

Factorial comparison: frame none versus SoG(p6); within each, point-only,
posterior_random, action_random, transport_random. Existing V5 none-frame runs
supply the exact matched controls, subject to numerical replay. New SoG arms
share a complete 20-epoch warmup including optimizer/scheduler/BatchNorm state.
120 epochs, batch32, same1126 train/119 development images, same seed17 first,
then29/43; scratch initialization, same3.097M parameters, no teacher, no phone
training, no new augmentation, unchanged V5 objective and field-weight schedule.
Source GT is expressed relative to the image anchor for the training losses;
reported predictions and recovery errors return to original camera RGB.
Reproduction error is invariant to simultaneous diagonal transformation of
prediction and GT. Risk scores stay in degrees. Invalid inputs stay refused.

Primary development contrasts: SoG transport versus SoG action/posterior and
point accuracy; interaction relative to corresponding V5 none-frame outcomes.
Report all seeds, best and final checkpoints, full mean and raw risk80; no
point-only untrained confidence claims. Reused development119 is not a fresh
benchmark. Any seed17 result is exploratory and cannot establish a contribution.
No test-winning checkpoint/head is selected. Already observed phone test data
cannot become validation for this new model: new confirmatory claims require
an additional locked disjoint external sample. Retain all negative outcomes.

Alternatives deferred: adding DINO teacher at the same time would confound the
mechanism; simple output averaging cannot test whether the internal frame helps.
After this factorial test, a separate teacher/no-teacher comparison may test
whether semantic pretraining improves the compact student's scene reasoning.
